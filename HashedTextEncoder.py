import binascii
import sys
import tensorflow_text as text

class HashedTextEncoder(tfds.features.text.TextEncoder):
  """Encodes text using PySuperFastHash"""

  def __init__(self):
    """Constructs HashedTextEncoder.
    Args:
      None
    """
  def encode(self, s):
    # Handle additional tokens
    s = tf.compat.as_text(s)
    s = s.lower()
    ids = []
    words = s.split(" ") 
    for substr in words:
      if not substr:
        continue
      newid = self.superFastHash(substr)
      ids.append(newid)
    return self.pad_incr(ids)

  def pad_incr(self,ids):
    """Add 1 to ids to account for pad."""
    return [i + 1 for i in ids]

  def decode(self, ids):
    raise NotImplementedError

  def load_from_file():
    raise NotImplementedError
    
  def save_to_file():
    raise NotImplementedError  
  
  def vocab_size():
    raise NotImplementedError  

  def get16bits(self, data):
    """Returns the first 16bits of a string"""
    return int(binascii.hexlify(data[1::-1]), 16)

  def superFastHash(self, data):
    # Start by stripping out UTF data
    data=data.encode("ascii","ignore")

    hash = length = len(data)
    if length == 0:
        return 0

    rem = length & 3
    length >>= 2

    while length > 0:
        hash += self.get16bits(data) & 0xFFFFFFFF
        tmp = (self.get16bits(data[2:])<< 11) ^ hash
        hash = ((hash << 16) & 0xFFFFFFFF) ^ tmp
        data = data[4:]
        hash += hash >> 11
        hash = hash & 0xFFFFFFFF
        length -= 1

    if rem == 3:
        hash += self.get16bits (data)
        hash ^= (hash << 16) & 0xFFFFFFFF
        hash ^= (data[2] << 18) & 0xFFFFFFFF
        hash += hash >> 11
    elif rem == 2:
        hash += self.get16bits (data)
        hash ^= (hash << 11) & 0xFFFFFFFF
        hash += hash >> 17
    elif rem == 1:
        hash += data[0]
        hash ^= (hash << 10) & 0xFFFFFFFF
        hash += hash >> 1

    hash = hash & 0xFFFFFFFF
    hash ^= (hash << 3) & 0xFFFFFFFF
    hash += hash >> 5
    hash = hash & 0xFFFFFFFF
    hash ^= (hash << 4) & 0xFFFFFFFF
    hash += hash >> 17
    hash = hash & 0xFFFFFFFF
    hash ^= (hash << 25) & 0xFFFFFFFF
    hash += hash >> 6

    #Shorter version throw away top bits
    hash = hash & 0xFFF

    return hash