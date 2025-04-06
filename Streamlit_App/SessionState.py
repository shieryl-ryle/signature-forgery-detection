
import streamlit as st

class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'

        """
        for key, val in kwargs.items():
            setattr(self, key, val)
    
    def __getattr__(self, item):
        # Get the value from st.session_state, if not available return None
        return st.session_state.get(item, None)

    def __setattr__(self, key, value):
        # Set the value to st.session_state
        st.session_state[key] = value


def get_session(**kwargs):
    """
    Gets a SessionState object for the current session.

    Creates a new object if necessary.

    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.
    """
    # Create a new SessionState object and save it in st.session_state if not already created
    if "_session_state" not in st.session_state:
        st.session_state["_session_state"] = SessionState(**kwargs)
    
    return st.session_state["_session_state"]