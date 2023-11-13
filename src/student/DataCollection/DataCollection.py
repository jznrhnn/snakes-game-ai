import shutil
import subprocess
import time
from datetime import datetime
from threading import Thread


def play():
    command = "java -jar AI-snake.jar negasnake.NegaSnake student.DataCollection"
    result = subprocess.run(
        command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print("process success")
        print(result.stdout)
    else:
        print("process failed")
        print(result.stderr)


def plays():
    # The number of threads to create
    number_of_threads = 10

    # Create the threads
    threads = []
    for i in range(number_of_threads):
        thread = Thread(target=play)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()


def backup():
    # Formats the time as YearMonthDayHourMinuteSecond
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    # Define the source and destination file paths
    src = 'DataCollection.csv'
    dst = f'DataCollection_{current_time}.csv'
    shutil.copyfile(src, dst)


for i in range(100):
    plays()
    backup()
    time.sleep(60)
