import argparse
import os

from googlesheet import GoogleSheetService, get_row_number

SAMPLE_SPREADSHEET_ID = '1aWqtGXlQOz_RpNFgCVJ9ACyQeDKOUVHDUafdbUnPzeY'
gss = None


def run_experiment(row):
    print(row.command)
    os.system(row.command)
    # raise Exception('sdf')


def main():
    print("Start")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmux', type=str)
    args = parser.parse_args()
    tmux_name = args.tmux

    gss = GoogleSheetService(SAMPLE_SPREADSHEET_ID)
    row = None
    for job in range(5):
        last_row = row
        row = gss.get_next_pending()
        if row is not None:
            row_number = get_row_number(row)
            gss.update_status('running', row_number)
            gss.update_tmux("{}_job:{}".format(tmux_name, str(job)), row_number)
            try:
                run_experiment(row)
            except:
                gss.update_status('failed', row_number)
            else:
                gss.update_status('finished', row_number)
        else:
            if last_row is not None:
                print("\n\n***** finished at job: {}\texperiment ID: {} *****\n\n"
                      "Last Experiment: ".format(job, last_row.Experiment))
                print(last_row)
            exit(0)


if __name__ == '__main__':
    main()
