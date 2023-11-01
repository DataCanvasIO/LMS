from subprocess import check_call, CalledProcessError


def prune(model_path, model_name, args, argv):
    command = ['lms_rt']
    command.extend(argv[1:])
    for index, segment in enumerate(command):
        if segment == "--model_path" or \
                segment == "--model_name":
            command[index] = "--model_path"
            command[index + 1] = model_path
            break
        elif segment.startswith("--model_path") or \
                segment.startswith("--model_name"):
            command[index] = "--model_path=" + model_path
            break
        else:
            continue
    try:
        check_call(" ".join(command), shell=True)
        print("Successfully pruned %s" % model_name)
    except CalledProcessError:
        pass
