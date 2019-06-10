import time, sys
barLength = 30 # Modify this to change the length of the progress bar

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block),'%3.3i'%(progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()

def progress_line(head,dat,iflush=True):
    text = "\r%s: %s [%s]"%(head,dat)
    sys.stdout.write(text)
    if iflush: sys.stdout.flush()

def convert_sec_to_string(second):
    if                     second<1.0e-9:
        time = '%8.3f [sec]'%second
    elif second>=1.e-9 and second<1.0e-6:
        time = '%3.0f [ ns]'%(second/1.0e-9)
    elif second>=1.e-6 and second<1.0e-3:
        time = '%3.0f [ us]'%(second/1.0e-6)
    elif second>=1.e-3 and second<1.0:
        time = '%3.0f [ ms]'%(second/1.0e-3)
    elif second>=1. and second<60:
        time = '%3.0f [sec]'%second
    elif second>=60. and second<3600:
        m = second/60.
        s = second - int(m)*60.
        time = '%2.2i [min] %2.2i [sec]'%(m,s)
    elif second>=3600:
        h = second/3600.
        m = (second - int(h)*3600.)/60.
        s = second - int(m) * 60. - int(h)*3600.
        time = '%i [hr] %2.2i [min]'%(h,m)

    time = '%17s'%time
    return time

def update_elapsed_time(second,head='Elapsed time',iflush=True):
    time = convert_sec_to_string(second)
    text = "\r%s: %s"%(head,time)
    sys.stdout.write(text)
    if iflush: sys.stdout.flush()
