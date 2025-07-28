import sys
import threading
import time
import itertools

# --- Simple Spinner Logic ---

# We use a global variable to tell the animation when to stop.
_spinning = False
_spinner_thread = None

def _animate():
    #The function that runs the actual animation.
    spinner_cycle = itertools.cycle(['|', '/', '-', '\\'])
    while _spinning:
        # This writes the animation to the screen.
        sys.stdout.write(f"\r{next(spinner_cycle)} Thinking...")
        sys.stdout.flush()
        time.sleep(0.1)
    # When done, this clears the "Thinking..." line.
    sys.stdout.write('\r' + ' ' * 15 + '\r')
    sys.stdout.flush()

def start_spinner():
    #Starts the spinner animation running in the background.
    global _spinning, _spinner_thread
    _spinning = True
    # We run the animation in a separate "thread" so it doesn't block our main app.
    _spinner_thread = threading.Thread(target=_animate)
    _spinner_thread.start()

def stop_spinner():
    #Stops the spinner animation.
    global _spinning
    _spinning = False
    if _spinner_thread:
        # Wait for the thread to finish cleaning up the line.
        _spinner_thread.join() 