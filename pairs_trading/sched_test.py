import sched, time
import datetime


def print_time(a='default'):
    print("From print_time", time.time(), a)

def print_some_times(s):
    print(time.time())
    s.enter(10, 1, print_time)
    s.enter(5, 2, print_time, argument=('positional',))
    s.enter(5, 1, print_time, kwargs={'a': 'keyword'})
    s.run()
    print(time.time())

def main():
    s = sched.scheduler(time.time, time.sleep)
    # print_some_times(s)

    print(time.time())

    s.enterabs(datetime.datetime.now() + datetime.timedelta(seconds=10), 1, print_time)

    s.run()

if __name__ == '__main__':
    main()
