from main_menu_0597 import Menu
from pygad_0597 import GeneticAlgorithm

from multiprocessing import Process, Queue


if __name__ == "__main__":

    main_menu = Menu()

    param_queue = Queue()
    ga_queue = Queue()
    menu_process = Process(target=main_menu.main_func, args=(param_queue,))

    menu_process.start()
    ga_list = []
    process_list = []

    ga_queue_list = []

    while True:
        data = param_queue.get()

        ga_data_list = [queue.get() for queue in ga_queue_list]
        for ga_data in ga_data_list:
            if ga_data == "FINISHED":  # Refreshing the list every time ga finishes training and sends not-empty queue
                main_menu.list_refresh()

        if data == "SHUTDOWN":  # Checking for shutdown command
            break

        val = data["fitness_params"]["Num_of_training_instances"]  # Amount of ga training instances
        for _ in range(0, val):
            if _ == 0:
                ga_queue_list = []  # Resetting the list with ga queues
                ga_flag = True

            ga_queue = Queue()
            ga = GeneticAlgorithm(data["fitness_params"], data["ga_params"])
            ga_process = Process(target=ga.ga_function, args=(ga_queue,))
            process_list.append(ga_process)
            ga_queue_list.append(ga_queue)
            ga_process.start()
