import time
import reminder as r

def time():
    time = time.time()
    time_array = pull_time()
    for times in time_array:
        if times < time:
            pull_reminder(times)
            # do_reminder()
            pass

def pull_reminder(time):
    for reminder in r.reminders:
        temp_time = reminder["time"]
        if temp_time == time:
            return reminder
time = []

def pull_time():
    time.clear()
    for reminder in r.reminders:
        temp_time = reminder["time"]
        time.append(temp_time)
    return time

print(pull_reminder("fhajhk"))