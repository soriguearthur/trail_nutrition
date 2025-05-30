import json
from datetime import datetime, timedelta

def float_hours_to_time(base_dt, hours_float):
    return base_dt + timedelta(hours=hours_float)

def create_event(dt, summary, description, uid_prefix):
    dtstart = dt.strftime("%Y%m%dT%H%M%S")
    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = f"{uid_prefix}-{dtstart}-{hash(description) % 10000}@example.com"

    return [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART;TZID=Europe/Paris:{dtstart}",
        f"SUMMARY:{summary}",
        f"DESCRIPTION:{description}",
        f"CATEGORIES:{'Nutrition' if 'Nutrition' in summary else 'Hydratation'}",
        "BEGIN:VALARM",
        "TRIGGER:-PT1M",
        f"DESCRIPTION:{description}",
        "ACTION:DISPLAY",
        "END:VALARM",
        "END:VEVENT"
    ]

def generate_ics(data, base_dt):
    ics = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "CALSCALE:GREGORIAN",
        "PRODID:-//Example Corp//Nutrition Plan//FR",
        "METHOD:PUBLISH",
        "BEGIN:VTIMEZONE",
        "TZID:Europe/Paris",
        "BEGIN:STANDARD",
        "DTSTART:20231029T030000",
        "TZOFFSETFROM:+0200",
        "TZOFFSETTO:+0100",
        "TZNAME:CET",
        "END:STANDARD",
        "END:VTIMEZONE"
    ]

    # Nutrition events
    for hours_str, action in data.get("nutrition", {}).items():
        hours = float(hours_str)
        event_dt = float_hours_to_time(base_dt, hours)
        ics.extend(create_event(event_dt, f"Nutrition : {action}", f"Nutrition : {action}", "nutrition"))

    # Hydratation events (une seule fois à l'heure de début)
    for hydra in data.get("hydratation", []):
        event_dt = float_hours_to_time(base_dt, hydra["heure"])
        summary = f"Hydratation : {hydra['nb_flasques']} flasques entre km {hydra['debut']} et {hydra['fin']}"
        description = f"Boire {hydra['nb_flasques']} flasques entre les km {hydra['debut']} et {hydra['fin']}"
        ics.extend(create_event(event_dt, summary, description, "hydratation"))
        
    ics.append("END:VCALENDAR")
    return "\n".join(ics)
