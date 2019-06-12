def finisher(finisher_queue, global_inference_counter, num_videos,
             job_id, termination_flag, sta_bar, fin_bar):
  import time
  import torch

  from control import TerminationFlag
  from rnb_logging import logcustom, TimeCardSummary
  from tqdm import tqdm

  time_card_summary = TimeCardSummary()
  torch.cuda.init()
  for i in range(8):
    with torch.cuda.device(i):
      insurance = torch.randn(1).cuda()

  tensors = {}
  sta_bar.wait()

  s = []
  counter = 0
  with torch.no_grad():
    with tqdm(total = num_videos) as progress_bar:
      while termination_flag.value == TerminationFlag.UNSET:
        tpl = finisher_queue.get()
        if tpl is None:
          break

        tensor, time_card = tpl
        time_card.record('finisher_start')
        if time_card.id not in tensors:
          tensors[time_card.id] = [None] * 4
        tensors[time_card.id][time_card.sub_id] = tensor
        tensors[time_card.id][time_card.sub_id+2] = time_card
        if not all(map(lambda x: x is not None, tensors[time_card.id])):
          continue

        tensor0, tensor1, tc0, tc1 = tensors[time_card.id]
        tensor = torch.cat([tensor0, tensor1])
        tc0.merge(tc1)
        tc0.record('finisher_finish')
        
        del tensors[time_card.id]
        time_card_summary.register(tc0)

        progress_bar.update(1)
        counter += 1
        if counter >= num_videos:
          print('Finished processing %d videos' % num_videos)
          termination_flag.value = TerminationFlag.TARGET_NUM_VIDEOS_REACHED

        # end = time.time()
        # s.append(end - start)

  fin_bar.wait()
  # print(sum(s))
  # print(len(s))

  NUM_SKIPS = 10
  time_card_summary.print_summary(NUM_SKIPS)
  with open(logcustom(job_id, 'timings.txt'), 'w') as f:
    time_card_summary.save_full_report(f)