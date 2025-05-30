import json
from datetime import datetime, timedelta

def create_event(dt, summary, description, uid_prefix):
    dtstart = dt.strftime("%Y%m%dT%H%M%S")
    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = f"{uid_prefix}-{dtstart}@example.com"

    return [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART;TZID=Europe/Paris:{dtstart}",
        f"SUMMARY:{summary}",
        f"DESCRIPTION:{description}",
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
    for hours_str, action in data["timing"].items():
        hours = float(hours_str)
        event_dt = float_hours_to_time(base_dt, hours)
        ics.extend(create_event(event_dt, f"Nutrition : {action}", f"Nutrition : {action}", "nutrition"))

    # Flasque events toutes les 30min entre 2 points
    distances = sorted(data["duree"].keys(), key=float)
    duree = data["duree"]
    flasques = data["flasques"]

    for i in range(len(distances) - 1):
        dist_start = distances[i]
        dist_end = distances[i + 1]

        if dist_end in flasques:
            nb_flasques = flasques[dist_end]
            t_start = duree[dist_start]
            t_end = duree[dist_end]
            t = t_start

            while t < t_end:
                event_dt = float_hours_to_time(base_dt, t)
                desc = f"Boire {nb_flasques} flasques entre les km {dist_start} et {dist_end}"
                ics.extend(create_event(event_dt, "Hydratation", desc, "flasque"))
                t += 0.5  # 30 min

    ics.append("END:VCALENDAR")
    return "\n".join(ics)
