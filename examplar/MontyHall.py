import random

score = 0;

strategy = 1  # do not switch
strategy = 2 # switch
N = 10000

for i in range(N):
    doors = [1,2,3]
    # placing the car randomly,
    car = random.randint(1,3)

    # player chooses a dor
    player = random.randint(1,3)

    #host open a door

    doors.remove(player)
    if car in doors:
        doors.remove(car)

    host = random.choice(doors)

    if strategy == 1:
        if player == car:
            score += 1
    else: #strategy = 2
        l = [1,2,3]
        l.remove(player)
        l.remove(host)
        if l[0] == car:
            score += 1
        

print(score/ N)
