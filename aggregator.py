def aggregator(aggregator_queue, shared_input_tensors, num_videos, gpus,
               job_id, termination_flag, sta_bar, fin_bar):
  import time
  import torch

  from control import TerminationFlag
  from rnb_logging import logcustom, TimeCardSummary
  from tqdm import tqdm

  time_card_summary = TimeCardSummary()
  torch.cuda.init()
  shape = shared_input_tensors[0][0].tensors[0].shape
  tensor_input_placeholders = []
  for i in gpus:
    with torch.cuda.device(i):
      device = torch.device('cuda:%d' % i)
      placeholder = torch.empty(*shape, dtype=torch.float32, device=device)
      tensor_input_placeholders.append(placeholder)

  results = {}

  sta_bar.wait()

  counter = 0
  with torch.no_grad():
    with tqdm(total=num_videos) as progress_bar:
      while termination_flag.value == TerminationFlag.UNSET:
        tpl = aggregator_queue.get()
        if tpl is None:
          break

        (signal, non_tensor_inputs), time_card = tpl
        time_card.record('aggregator_start')

        if signal is not None:
          instance_idx, tensor_idx = signal
          tensor_event = shared_input_tensors[instance_idx][tensor_idx]

          if tensor_event.event.is_set() and \
              termination_flag.value != TerminationFlag.UNSET:
            break

          placeholder = tensor_input_placeholders[instance_idx]
          placeholder.copy_(tensor_event.tensors[0])

          tensor_event.event.set()

        tmp = placeholder.cpu().numpy().sum(axis=0)
        if time_card.id not in results:
          results[time_card.id] = (tmp, time_card)
          continue
        else:
          prev_tmp, prev_time_card = results.pop(time_card.id)
          final_tmp = prev_tmp + tmp
          final = final_tmp.argmax()
          time_card.merge(prev_time_card)



        time_card.record('aggregator_finish')
        time_card_summary.register(time_card)

        progress_bar.update(1)
        counter += 1
        if counter >= num_videos:
          print('Finished processing %d videos' % num_videos)
          termination_flag.value = TerminationFlag.TARGET_NUM_VIDEOS_REACHED

      if shared_input_tensors is not None:
        for instance_tensors in shared_input_tensors:
          for protected_tensor in instance_tensors:
            protected_tensor.event.set()

  fin_bar.wait()

  NUM_SKIPS = 10
  time_card_summary.print_summary(NUM_SKIPS)
  with open(logcustom(job_id, 'timings.txt'), 'w') as f:
    time_card_summary.save_full_report(f)
