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
