# -*- coding:utf-8 -*-
# @author :adolf
import time


def timeit(func):
    def wrap():
        start = time.time()
        func()
        end = time.time()
        print("foo is used:", end - start)

    return wrap

@timeit
def test():
    print('in test')
    time.sleep(1)


test()
