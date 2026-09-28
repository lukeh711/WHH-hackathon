import json

file = open("reminders.json", "r")
reminders = json.load(file)
file.close()

time = []

def pull_time():
    time.clear()
    for reminder in reminders:
        temp_time = reminder["time"]
        time.append(temp_time)
    print(time)

def add_reminder(text, time):
    new_reminder = {"text": text, "time": time}
    reminders.append(new_reminder)
    pull_time()

def remove_reminder(text):
    texts = []
    for i in range(len(reminders)):
        temp_reminder = reminders[i]["text"]
        texts.append(temp_reminder)
    if text in texts:
        reminders.remove(reminders[i])
    pull_time()

def close(reminders):
    file = open("reminders.json", "w")
    reminders_json = json.dumps(reminders)
    file.write(reminders_json)
    file.close()

remove_reminder("fjasjfhjdk")
add_reminder("fjasjfhjdk", "fhajhk")
close(reminders)