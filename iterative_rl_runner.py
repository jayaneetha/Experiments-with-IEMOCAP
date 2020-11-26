import argparse
import os
import time

from googlesheet import GoogleSheetService, get_row_number

SAMPLE_SPREADSHEET_ID = '1aWqtGXlQOz_RpNFgCVJ9ACyQeDKOUVHDUafdbUnPzeY'
gss = None


def run_experiment(row, command_suffix: str):
    command = row.command + " " + command_suffix
    print(command)
    os.system(command)
    # raise Exception('sdf')


def process(row, tmux, job, suffix):
    row_number = get_row_number(row)
    gss.update_status('running', row_number)
    gss.update_tmux("{}_job:{}".format(tmux, str(job)), row_number)
    try:
        run_experiment(row, suffix)
    except:
        gss.update_status('failed', row_number)
    else:
        gss.update_status('finished', row_number)


def main():
    print("Start")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmux', type=str, default='debug')
    parser.add_argument('--num-jobs', type=int, default=-1)
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()
    tmux_name = args.tmux
    num_jobs = args.num_jobs

    gss = GoogleSheetService(SAMPLE_SPREADSHEET_ID)
    row = None

    if args.num_jobs > -1:
        for job in range(num_jobs):
            last_row = row
            row = gss.get_next_pending()
            if row is not None:
                process(row, tmux_name, job, args.suffix)
            else:
                if last_row is not None:
                    print("\n\n***** finished at job: {}\texperiment ID: {} *****\n\n"
                          "Last Experiment: ".format(job, last_row.Experiment))
                    print(last_row)
                exit(0)
    else:
        print("Hoping to endless mode")
        job = 0
        while True:
            row = gss.get_next_pending()
            if row is not None:
                process(row, tmux_name, job, args.suffix)
                job += 1
            else:
                print("sleeping for 120s")
                time.sleep(120)


if __name__ == '__main__':
    main()
