def clear_speech_queue():

    while not speech_queue.empty():

        try:
            speech_queue.get_nowait()
            speech_queue.task_done()

        except queue.Empty:
            break