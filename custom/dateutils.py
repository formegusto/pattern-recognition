def day_to_kr(weekday):
    kr = "월"

    if weekday == 0:
        kr = "월"
    elif weekday == 1:
        kr = "화"
    elif weekday == 2:
        kr = "수"
    elif weekday == 3:
        kr = "목"
    elif weekday == 4:
        kr = "금"
    elif weekday == 5:
        kr = "토"
    elif weekday == 6:
        kr = "일"

    return kr


def kr_to_day(day_kr):
    day = 0

    if day_kr == "월":
        day = 0
    elif day_kr == "화":
        day = 1
    elif day_kr == "수":
        day = 2
    elif day_kr == "목":
        day = 3
    elif day_kr == "금":
        day = 4
    elif day_kr == "토":
        day = 5
    elif day_kr == "일":
        day = 6

    return day
