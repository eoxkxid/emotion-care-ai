start_hour = int(input("시간:"))
if start_hour < 0 or start_hour > 23:
    print("Invalid hour. Please enter a value between 0 and 23.")
    exit(1)
start_minute = int(input("분:"))
if start_minute < 0 or start_minute > 59:
    print("Invalid minute. Please enter a value between 0 and 59.")
    exit(1)
start_second = int(input("초:"))
if start_second < 0 or start_second > 59:
    print("Invalid second. Please enter a value between 0 and 59.")
    exit(1)

print(start_hour,":", start_minute,":", start_second)
plus_seconds = int(input("구울시간 (초단위):"))

start_second += plus_seconds
if start_second >= 60:
    start_minute += start_second // 60
    start_second = start_second % 60   
if start_minute >= 60:
    start_hour += start_minute // 60
    start_minute = start_minute % 60
if start_hour >= 24:
    start_hour = start_hour % 24   

print(f"{start_hour:02}:{start_minute:02}:{start_second:02}")

 