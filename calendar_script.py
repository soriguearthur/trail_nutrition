import json
from datetime import datetime, timedelta

def float_hours_to_time(base_dt, hours_float):
    return base_dt + timedelta(hours=hours_float)

def generate_ics(timing_dict, base_dt):
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

    for hours_str, action in timing_dict.items():
        hours = float(hours_str)
        event_dt = float_hours_to_time(base_dt, hours)
        dtstart = event_dt.strftime("%Y%m%dT%H%M%S")
        dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        uid = f"{action}-{hours_str}@example.com"

        ics.extend([
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{dtstamp}",
            f"DTSTART;TZID=Europe/Paris:{dtstart}",
            f"SUMMARY:Take {action}",
            f"DESCRIPTION:Nutrition reminder: {action}",
            "BEGIN:VALARM",
            "TRIGGER:-PT1M",
            "DESCRIPTION:Reminder",
            "ACTION:DISPLAY",
            "END:VALARM",
            "END:VEVENT"
        ])

    ics.append("END:VCALENDAR")

    return "\n".join(ics)


# start_date = datetime(2025, 5, 30, 7, 0, 0)  # 30 mai 2025, 07:00
# with open('data.json', 'r') as f:
#     timing_json = json.load(f)['timing']
# ics_content = generate_ics(timing_json, start_date)
# with open("nutrition_plan.ics", "w") as f:
#     f.write(ics_content)
