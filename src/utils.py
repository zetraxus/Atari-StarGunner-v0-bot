def print_logs(save_to_file, path, info):
    if save_to_file:
        with open(path, 'a+') as f:
            f.write(info + '\n')
    else:
        print(info)
