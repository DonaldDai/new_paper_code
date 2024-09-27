from multiprocessing import Pool, cpu_count
import os
import time

def sub_time_task(i):
    print('子子进程: {} - 任务{}'.format(os.getpid(), i))
    time.sleep(2)
    print("子子进程结果: {}".format(8 ** 20))

def long_time_task(i):
    print('子进程: {} - 任务{}'.format(os.getpid(), i))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))
    for i in range(10):
        sub_time_task(i)

def main():
    print("CPU内核数:{}".format(cpu_count()))
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('等待所有子进程完成。')
    p.close()
    p.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))

if __name__=='__main__':
    main()