from halo import Halo

# --- Halo Spinner Logic ---

# Global spinner instance
_spinner = None

def start_spinner():
    """Starts a beautiful spinner animation."""
    global _spinner
    _spinner = Halo(text='Thinking...', spinner='bouncingBar')
    _spinner.start()

def stop_spinner():
    """Stops the spinner animation."""
    global _spinner
    if _spinner:
        _spinner.stop()
        _spinner = None 