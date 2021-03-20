import scipy.spatial.distance as distance


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def get_entry_str(change_entry, dist=None, change_seq=False, score=None):
    entry_str = ''
    entry_str += 'Id: %s\n' % (change_entry.id)
    entry_str += 'Prev:\n%s\nAfter:\n%s\n' % (change_entry.untokenized_previous_code_chunk, 
                                              change_entry.untokenized_updated_code_chunk)
    if dist is not None:
        entry_str += 'Distance: %f\n' % dist
        
    if change_seq:
        entry_str += 'Change Sequence: %s\n' % change_entry.change_seq
        
    entry_str += f"Score: {score if score is not None else '#'}\n"
    
    entry_str += '*' * 5
    
    return entry_str

