import base64
import hashlib
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import TemplateNotFound
from starlette.middleware.sessions import SessionMiddleware

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "anpr_mvp.db"
UPLOAD_DIR = BASE_DIR / "uploads"
RETENTION_DAYS = 180
MAX_VEHICLES_PER_TRIGGER = 4
SECRET_KEY = os.environ.get("ANPR_SECRET", secrets.token_hex(24))

app = FastAPI(title="ANPR Guard MVP")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def format_event_time(value: str) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value).replace("T", " ").split("+")[0].split(".")[0]


templates.env.filters["event_time"] = format_event_time


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    conn = db_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('guard','admin'))
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_ai TEXT,
            plate_final TEXT,
            ai_conf REAL,
            direction TEXT,
            gate TEXT,
            vehicle_type TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            event_time TEXT NOT NULL,
            best_image_path TEXT,
            all_images_json TEXT NOT NULL,
            reviewed_by TEXT,
            reviewed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    for column_name, column_sql in {
        "parent_event_id": "ALTER TABLE events ADD COLUMN parent_event_id INTEGER",
        "sub_event_no": "ALTER TABLE events ADD COLUMN sub_event_no INTEGER",
    }.items():
        existing = [row[1] for row in cur.execute("PRAGMA table_info(events)").fetchall()]
        if column_name not in existing:
            cur.execute(column_sql)

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    # seed accounts
    users = cur.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    if users == 0:
        cur.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)", ("guard1", hash_password("guard123"), "guard"))
        cur.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)", ("admin1", hash_password("admin123"), "admin"))

    conn.commit()
    conn.close()


def add_audit(conn, event_id: Optional[int], actor: str, action: str, details: str = ""):
    conn.execute(
        "INSERT INTO audit_logs(event_id,actor,action,details,created_at) VALUES(?,?,?,?,?)",
        (event_id, actor, action, details, utcnow_iso()),
    )


def require_user(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    return user


def is_valid_admin_password(conn: sqlite3.Connection, password: str) -> bool:
    if not password:
        return False
    pw_hash = hash_password(password)
    row = conn.execute(
        "SELECT id FROM users WHERE role='admin' AND password_hash=? LIMIT 1",
        (pw_hash,),
    ).fetchone()
    return row is not None



def parse_event_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def is_unread_or_unrecognized(value: str) -> bool:
    text = (value or "").strip().upper()
    if not text:
        return True
    unread_tokens = ("NOT_FOUND", "PLATE_NOT_CLEAR", "UNREAD", "UNRECOGNIZED", "UNKNOWN", "NO_PLATE")
    return text.startswith("??") or any(token in text for token in unread_tokens)


def unread_event_count(conn: sqlite3.Connection) -> int:
    rows = conn.execute("SELECT plate_ai FROM events WHERE status='pending'").fetchall()
    return sum(1 for row in rows if is_unread_or_unrecognized(row["plate_ai"]))


def common_context(conn: sqlite3.Connection) -> dict:
    return {"unread_alert_count": unread_event_count(conn)}


def safe_return_path(value: str) -> str:
    if not value or not value.startswith("/") or value.startswith("//"):
        return "/dashboard"
    return value


def redirect_with_incident_error(return_to: str, message: str):
    target = safe_return_path(return_to)
    separator = "&" if "?" in target else "?"
    return RedirectResponse(f"{target}{separator}incident_error={quote(message)}", status_code=303)


def ensure_parent_allows_sub_event(parent) -> Optional[str]:
    event_dt = parse_event_datetime(parent["event_time"])
    if not event_dt:
        return "Original event time is invalid; cannot add incident safely."
    age_seconds = (datetime.now(timezone.utc) - event_dt).total_seconds()
    if age_seconds > 4 * 60 * 60:
        return "Sub-events must be added within 4 hours of the original trigger event."
    return None


def save_upload_file(upload: Optional[UploadFile]) -> Optional[str]:
    if not upload or not upload.filename:
        return None
    suffix = Path(upload.filename).suffix or ".jpg"
    out_name = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
    out_path = UPLOAD_DIR / out_name
    out_path.write_bytes(upload.file.read())
    return f"uploads/{out_name}"

def cleanup_non_best_images(image_paths: list[str], best_image: Optional[str]):
    for p in image_paths:
        if best_image and p == best_image:
            continue
        full = BASE_DIR / p
        if full.exists() and full.is_file():
            full.unlink(missing_ok=True)


def parse_images_json(raw_value) -> list[str]:
    """Safely parse image list from DB, including legacy/bad rows."""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [x for x in raw_value if x]
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return []
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [x for x in parsed if x]
            if isinstance(parsed, str):
                return [parsed]
            return []
        except Exception:
            # tolerate older plain-string path format
            return [raw_value]
    return []


def get_root_event_id(event) -> int:
    return int(event["parent_event_id"] or event["id"])


def create_manual_sub_event(
    conn: sqlite3.Connection,
    parent_id: int,
    plate: str,
    actor: str,
    fallback_event,
    best_image: Optional[str],
    uploaded_image_path: Optional[str] = None,
    enforce_four_hour_window: bool = True,
):
    parent = conn.execute("SELECT * FROM events WHERE id=?", (parent_id,)).fetchone()
    if not parent:
        return None, "Original trigger event ID was not found."
    if parent["parent_event_id"]:
        return None, "Please enter the original trigger event ID, not another sub-event ID."
    if enforce_four_hour_window:
        time_error = ensure_parent_allows_sub_event(parent)
        if time_error:
            return None, time_error

    current_count = conn.execute(
        "SELECT COUNT(*) AS c FROM events WHERE parent_event_id=?",
        (parent_id,),
    ).fetchone()["c"]
    if current_count >= (MAX_VEHICLES_PER_TRIGGER - 1):
        return None, f"This trigger already has the maximum {MAX_VEHICLES_PER_TRIGGER} vehicles."

    plate = plate.strip().upper()
    if not plate:
        return None, "Enter the extra vehicle plate before adding a sub-event."

    parent_images = parse_images_json(parent["all_images_json"])
    fallback_images = parse_images_json(fallback_event["all_images_json"])
    chosen_best = uploaded_image_path or (best_image if best_image in fallback_images else (parent["best_image_path"] or (parent_images[0] if parent_images else None)))
    images_json = json.dumps([chosen_best] if chosen_best else [])
    now = utcnow_iso()
    sub_no = int(current_count) + 1

    cur = conn.execute(
        """
        INSERT INTO events(plate_ai,plate_final,ai_conf,direction,gate,vehicle_type,status,event_time,best_image_path,all_images_json,reviewed_by,reviewed_at,created_at,updated_at,parent_event_id,sub_event_no)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "MANUAL",
            plate,
            1.0,
            parent["direction"] or fallback_event["direction"],
            parent["gate"] or fallback_event["gate"],
            fallback_event["vehicle_type"] or parent["vehicle_type"] or "vehicle",
            "approved",
            parent["event_time"],
            chosen_best,
            images_json,
            actor,
            now,
            now,
            now,
            parent_id,
            sub_no,
        ),
    )
    sub_id = cur.lastrowid
    add_audit(conn, sub_id, actor, "add_sub_event", f"parent={parent_id}; plate={plate}; sub_no={sub_no}")
    add_audit(conn, parent_id, actor, "add_sub_event", f"sub_event={sub_id}; plate={plate}; sub_no={sub_no}")
    return sub_id, None


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = db_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()

    if not row or row["password_hash"] != hash_password(password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    request.session["user"] = {"username": row["username"], "role": row["role"]}
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, incident_error: str = ""):
    user = require_user(request)
    conn = db_conn()
    rows = conn.execute(
        "SELECT * FROM events WHERE status='pending' ORDER BY datetime(event_time) DESC LIMIT 100"
    ).fetchall()
    context = {"request": request, "user": user, "events": rows, "incident_error": incident_error, **common_context(conn)}
    conn.close()
    return templates.TemplateResponse("dashboard.html", context)


@app.get("/events/{event_id}", response_class=HTMLResponse)
def event_detail(request: Request, event_id: int, source: str = "dashboard", incident_error: str = ""):
    user = require_user(request)
    conn = db_conn()
    event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
    if not event:
        conn.close()
        raise HTTPException(404, "Event not found")
    images = parse_images_json(event["all_images_json"])
    root_event_id = get_root_event_id(event)
    sub_events = conn.execute(
        "SELECT * FROM events WHERE parent_event_id=? ORDER BY sub_event_no ASC, id ASC",
        (root_event_id,),
    ).fetchall()
    base_context = common_context(conn)
    conn.close()
    source = "logs" if source == "logs" else "dashboard"
    context = {
        "request": request,
        "user": user,
        "event": event,
        "images": images,
        "source": source,
        "error": None,
        "message": None,
        "incident_error": incident_error,
        "root_event_id": root_event_id,
        "sub_events": sub_events,
        "sub_event_slots_left": max(0, (MAX_VEHICLES_PER_TRIGGER - 1) - len(sub_events)),
        **base_context,
    }
    try:
        return templates.TemplateResponse("event_detail.html", context)
    except TemplateNotFound:
        # Compatibility fallback if local projects use `event.html`.
        return templates.TemplateResponse("event.html", context)


@app.post("/events/{event_id}/review")
def review_event(
    request: Request,
    event_id: int,
    action: str = Form(...),
    corrected_plate: str = Form(""),
    best_image: str = Form(""),
    source: str = Form("dashboard"),
    admin_password: str = Form(""),
    sub_parent_id: str = Form(""),
    sub_plate: str = Form(""),
):
    user = require_user(request)
    conn = db_conn()
    event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
    if not event:
        conn.close()
        raise HTTPException(404, "Event not found")

    images = parse_images_json(event["all_images_json"])
    final_plate = corrected_plate.strip() or event["plate_ai"]
    status_map = {
        "approve": "approved",
        "edit_save": "approved",
        "reject": "rejected",
        "flag": "flagged",
    }
    source = "logs" if source == "logs" else "dashboard"

    if action == "add_sub_event":
        if source == "logs":
            conn.close()
            raise HTTPException(400, "Sub-events can only be added from dashboard review.")
        try:
            parent_id = int(sub_parent_id or get_root_event_id(event))
        except ValueError:
            parent_id = 0
        sub_id, sub_error = create_manual_sub_event(
            conn,
            parent_id,
            sub_plate,
            user["username"],
            event,
            best_image,
        )
        if sub_error:
            root_event_id = get_root_event_id(event)
            sub_events = conn.execute(
                "SELECT * FROM events WHERE parent_event_id=? ORDER BY sub_event_no ASC, id ASC",
                (root_event_id,),
            ).fetchall()
            context = {
                "request": request,
                "user": user,
                "event": event,
                "images": images,
                "source": "dashboard",
                "error": sub_error,
                "message": None,
                "root_event_id": root_event_id,
                "sub_events": sub_events,
                "sub_event_slots_left": max(0, (MAX_VEHICLES_PER_TRIGGER - 1) - len(sub_events)),
                **common_context(conn),
            }
            conn.close()
            return templates.TemplateResponse("event_detail.html", context, status_code=400)
        conn.commit()
        conn.close()
        return RedirectResponse(f"/events/{event_id}?source=dashboard", status_code=303)

    if source == "logs":
        # Historical logs page should only allow plate corrections.
        if action != "edit_save":
            conn.close()
            raise HTTPException(400, "Only Edit & Save is allowed from historical logs")
        if not is_valid_admin_password(conn, admin_password):
            context = {
                "request": request,
                "user": user,
                "event": event,
                "images": images,
                "source": "logs",
                "error": "Admin password required to update a saved historical record.",
                "message": None,
                "root_event_id": get_root_event_id(event),
                "sub_events": conn.execute(
                    "SELECT * FROM events WHERE parent_event_id=? ORDER BY sub_event_no ASC, id ASC",
                    (get_root_event_id(event),),
                ).fetchall(),
                "sub_event_slots_left": 0,
                **common_context(conn),
            }
            conn.close()
            return templates.TemplateResponse("event_detail.html", context, status_code=403)

    status = status_map.get(action)
    if not status:
        conn.close()
        raise HTTPException(400, "Invalid action")

    if best_image and best_image in images:
        chosen_best = best_image
    else:
        chosen_best = event["best_image_path"] or (images[0] if images else None)

    kept_images_json = json.dumps([chosen_best] if (status == "approved" and chosen_best) else images)

    conn.execute(
        """
        UPDATE events
        SET status=?, plate_final=?, best_image_path=?, all_images_json=?, reviewed_by=?, reviewed_at=?, updated_at=?
        WHERE id=?
        """,
        (status, final_plate, chosen_best, kept_images_json, user["username"], utcnow_iso(), utcnow_iso(), event_id),
    )

    if status == "approved" and chosen_best and images:
        conn.execute(
            "UPDATE events SET best_image_path=?, all_images_json=?, updated_at=? WHERE parent_event_id=? AND best_image_path IN ("
            + ",".join("?" for _ in images)
            + ")",
            (chosen_best, json.dumps([chosen_best]), utcnow_iso(), event_id, *images),
        )

    add_audit(conn, event_id, user["username"], action, f"final={final_plate}; best_image={chosen_best}")
    next_pending = conn.execute(
        "SELECT id FROM events WHERE status='pending' AND id != ? ORDER BY datetime(event_time) ASC LIMIT 1",
        (event_id,),
    ).fetchone()
    conn.commit()
    conn.close()

    if status == "approved":
        cleanup_non_best_images(images, chosen_best)

    if source == "logs":
        return RedirectResponse("/logs", status_code=303)
    if next_pending:
        return RedirectResponse(f"/events/{next_pending['id']}", status_code=303)
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/logs", response_class=HTMLResponse)
def logs(
    request: Request,
    q: str = "",
    direction: str = "",
    status: str = "",
    reviewer: str = "",
    gate: str = "",
    event_date: str = "",
    time_from: str = "",
    time_to: str = "",
):
    user = require_user(request)
    conn = db_conn()

    sql = "SELECT * FROM events WHERE 1=1"
    params = []

    if q:
        sql += " AND (plate_final LIKE ? OR plate_ai LIKE ? OR vehicle_type LIKE ?)"
        params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
    if direction:
        sql += " AND direction=?"
        params.append(direction)
    if status:
        sql += " AND status=?"
        params.append(status)
    if reviewer:
        sql += " AND reviewed_by LIKE ?"
        params.append(f"%{reviewer}%")
    if gate:
        sql += " AND gate LIKE ?"
        params.append(f"%{gate}%")
    if event_date:
        sql += " AND date(event_time)=date(?)"
        params.append(event_date)
    if time_from:
        tf = time_from if len(time_from) == 8 else f"{time_from}:00"
        sql += " AND substr(replace(event_time,'T',' '),12,8) >= ?"
        params.append(tf)
    if time_to:
        tt = time_to if len(time_to) == 8 else f"{time_to}:59"
        sql += " AND substr(replace(event_time,'T',' '),12,8) <= ?"
        params.append(tt)

    sql += " ORDER BY datetime(event_time) DESC LIMIT 300"
    rows = conn.execute(sql, params).fetchall()
    base_context = common_context(conn)
    conn.close()

    return templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
            "user": user,
            "events": rows,
            **base_context,
            "filters": {
                "q": q,
                "direction": direction,
                "status": status,
                "reviewer": reviewer,
                "gate": gate,
                "event_date": event_date,
                "time_from": time_from,
                "time_to": time_to,
            },
        },
    )


@app.post("/events/manual-incident")
def manual_incident(
    request: Request,
    parent_event_id: int = Form(...),
    sub_plate: str = Form(""),
    return_to: str = Form("/dashboard"),
    incident_image: Optional[UploadFile] = File(None),
):
    user = require_user(request)
    conn = db_conn()
    parent = conn.execute("SELECT * FROM events WHERE id=?", (parent_event_id,)).fetchone()
    if not parent:
        conn.close()
        return redirect_with_incident_error(return_to, "Original trigger event was not found.")
    if parent["parent_event_id"]:
        conn.close()
        return redirect_with_incident_error(return_to, "Please enter the original trigger event ID, not another sub-event ID.")
    if not sub_plate.strip():
        conn.close()
        return redirect_with_incident_error(return_to, "Enter the incident plate or note before saving.")
    time_error = ensure_parent_allows_sub_event(parent)
    if time_error:
        conn.close()
        return redirect_with_incident_error(return_to, time_error)

    uploaded_path = save_upload_file(incident_image)
    sub_id, sub_error = create_manual_sub_event(
        conn,
        parent_event_id,
        sub_plate,
        user["username"],
        parent,
        None,
        uploaded_image_path=uploaded_path,
    )
    if sub_error:
        conn.close()
        return redirect_with_incident_error(return_to, sub_error)

    conn.commit()
    conn.close()
    return RedirectResponse(safe_return_path(return_to), status_code=303)


@app.post("/api/events")
async def api_receive_event(
    request: Request,
    plate_ai: str = Form(""),
    ai_conf: float = Form(0.0),
    direction: str = Form(""),
    gate: str = Form(""),
    vehicle_type: str = Form(""),
    event_time: str = Form(""),
    files: list[UploadFile] = File(default=[]),
):
    # simple bearer token option
    token = request.headers.get("x-api-key", "")
    expected = os.environ.get("ANPR_API_KEY", "")
    if expected and token != expected:
        raise HTTPException(401, "Invalid API key")

    image_paths = []
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        payload = await request.json()
        # Backward compatible field aliases from older client payloads.
        plate_ai = payload.get("plate_ai") or payload.get("plate_number") or plate_ai
        ai_conf = float(payload.get("ai_conf") or payload.get("confidence") or (ai_conf or 0.0))
        direction = payload.get("direction", direction)
        gate = payload.get("gate", gate)
        vehicle_type = payload.get("vehicle_type", vehicle_type)
        event_time = payload.get("event_time", event_time)
        images_b64 = payload.get("images_base64", []) or []

        for encoded in images_b64:
            try:
                data = base64.b64decode(encoded)
            except Exception:
                continue
            out_name = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            out_path = UPLOAD_DIR / out_name
            out_path.write_bytes(data)
            image_paths.append(f"uploads/{out_name}")
    else:
        for f in files:
            suffix = Path(f.filename).suffix or ".jpg"
            out_name = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
            out_path = UPLOAD_DIR / out_name
            content = await f.read()
            out_path.write_bytes(content)
            image_paths.append(f"uploads/{out_name}")

    if not event_time:
        event_time = utcnow_iso()

    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO events(plate_ai,plate_final,ai_conf,direction,gate,vehicle_type,status,event_time,best_image_path,all_images_json,reviewed_by,reviewed_at,created_at,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            plate_ai,
            None,
            ai_conf,
            direction,
            gate,
            vehicle_type,
            "pending",
            event_time,
            image_paths[0] if image_paths else None,
            json.dumps(image_paths),
            None,
            None,
            utcnow_iso(),
            utcnow_iso(),
        ),
    )
    event_id = cur.lastrowid
    add_audit(conn, event_id, "anpr_api", "ingest", f"plate_ai={plate_ai}; images={len(image_paths)}")
    conn.commit()
    conn.close()
    return {"ok": True, "event_id": event_id}


@app.post("/api/sync-actions")
async def sync_actions(request: Request):
    user = require_user(request)
    payload = await request.json()
    actions = payload.get("actions", [])
    applied = 0

    conn = db_conn()
    for item in actions:
        event_id = int(item.get("event_id", 0))
        action = item.get("action", "")
        corrected_plate = (item.get("corrected_plate") or "").strip()
        best_image = (item.get("best_image") or "").strip()
        source = "logs" if item.get("source") == "logs" else "dashboard"
        admin_password = (item.get("admin_password") or "").strip()

        event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            continue

        images = parse_images_json(event["all_images_json"])
        if action == "add_sub_event":
            if source == "logs":
                continue
            try:
                parent_id = int(item.get("sub_parent_id") or get_root_event_id(event))
            except ValueError:
                parent_id = 0
            sub_id, sub_error = create_manual_sub_event(
                conn,
                parent_id,
                item.get("sub_plate") or "",
                user["username"],
                event,
                best_image,
            )
            if not sub_error:
                applied += 1
            continue

        status_map = {"approve": "approved", "edit_save": "approved", "reject": "rejected", "flag": "flagged"}
        status = status_map.get(action)
        if not status:
            continue
        if source == "logs":
            if action != "edit_save":
                continue
            if not is_valid_admin_password(conn, admin_password):
                continue

        chosen_best = best_image if best_image in images else (event["best_image_path"] or (images[0] if images else None))
        final_plate = corrected_plate or event["plate_ai"]

        kept_images_json = json.dumps([chosen_best] if (status == "approved" and chosen_best) else images)
        conn.execute(
            "UPDATE events SET status=?, plate_final=?, best_image_path=?, all_images_json=?, reviewed_by=?, reviewed_at=?, updated_at=? WHERE id=?",
            (status, final_plate, chosen_best, kept_images_json, user["username"], utcnow_iso(), utcnow_iso(), event_id),
        )
        add_audit(conn, event_id, user["username"], f"offline_{action}", f"final={final_plate}; best={chosen_best}")
        if status == "approved":
            cleanup_non_best_images(images, chosen_best)
        applied += 1

    conn.commit()
    conn.close()
    return JSONResponse({"ok": True, "applied": applied})


@app.get("/api/audit/{event_id}")
def audit_for_event(request: Request, event_id: int):
    require_user(request)
    conn = db_conn()
    rows = conn.execute("SELECT * FROM audit_logs WHERE event_id=? ORDER BY created_at ASC", (event_id,)).fetchall()
    conn.close()
    return {"event_id": event_id, "logs": [dict(r) for r in rows]}
