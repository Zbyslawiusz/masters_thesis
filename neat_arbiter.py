from neat_menu import Menu
from neat_algorithm import NeatAlgorithm

from multiprocessing import Process, Queue


if __name__ == "__main__":

    main_menu = Menu()

    param_queue = Queue()
    neat_queue = Queue()
    menu_process = Process(target=main_menu.main_func, args=(param_queue,))

    menu_process.start()
    neat_list = []
    process_list = []

    neat_queue_list = []

    while True:
        data = param_queue.get()

        neat_data_list = [queue.get() for queue in neat_queue_list]
        for neat_data in neat_data_list:
            # Refreshing the list every time NEAT finishes training and sends not-empty queue
            if neat_data == "FINISHED":
                main_menu.list_refresh()

        if data == "SHUTDOWN":  # Checking for shutdown command
            for process in process_list:
                process.terminate()
            break

        val = data["fitness_params"]["Num_of_training_instances"]  # Amount of NEAT training instances
        for _ in range(0, val):
            if _ == 0:
                neat_queue_list = []  # Resetting the list with NEAT queues
                neat_flag = True

            neat_queue = Queue()
            # ga = NeatAlgorithm(data["fitness_params"], data["ga_params"], data["config_file"])
            neat = NeatAlgorithm(data["fitness_params"], data["neat_params"])
            neat_process = Process(target=neat.neat_start, args=(neat_queue,))
            process_list.append(neat_process)
            neat_queue_list.append(neat_queue)
            neat_process.start()
