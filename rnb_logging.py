import os

def logname(beta, g, g_idx, r, r_idx, b):
  if not os.path.isdir('logs'):
    os.mkdir('logs')
  return 'logs/expo%03d-g%d-%d-r%d-%d-b%d' % (beta, g, g_idx, r, r_idx, b)
