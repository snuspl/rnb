import os

def logroot(job_id):
  path = 'logs/%s' % job_id
  os.makedirs(path, exist_ok=True)
  return path


def logmeta(job_id):
  root = logroot(job_id)
  return '%s/log-meta.txt' % root


def logname(job_id, g_idx, r_idx):
  root = logroot(job_id)
  return '%s/g%d-r%d.txt' % (root, g_idx, r_idx)
