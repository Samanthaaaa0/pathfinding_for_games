import subprocess, os,glob, time,json,sys,requests

N=int(sys.argv[2])
processPool=[]


def remove_first_line(input_file, output_file):
    # Check if the input file is not empty
    try:
        with open(input_file, 'r') as infile:
            first_char = infile.read(1)
            if not first_char:
                print(f"The file '{input_file}' is empty.")
                return
            # Move the file pointer back to the beginning
            infile.seek(0)
            data = infile.read().splitlines(True)
    except FileNotFoundError:
        print(f"The file '{input_file}' does not exist.")
        return

    # Write to the output file
    with open(output_file, 'w') as outfile:
        outfile.writelines(data[1:])
    return data[0]

#data=remove_first_line('./input.txt', './errors.txt')
with open(sys.argv[1],"r") as f:
    CMDPOOL=[l for l in f]

errors=[]

for data in CMDPOOL:
    cmd=data.strip().split(" ")
    print(subprocess.list2cmdline(cmd))

    if (len(processPool)>=N):
        finish = False
        while not finish:
            time.sleep(1)
            for p in range(0,len(processPool)):
                if p >= len(processPool):
                    break
                result=processPool[p].poll()
                if result is not None:
                    if result!=0:
                        
                        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                        print(processPool[p].args)
                        print(processPool[p].stderr)
                        errors.append(' '.join(processPool[p].args))
                        #with open("errors.txt",'a') as f:
                        #    print(' '.join(processPool[p].args),file=f)
                    processPool.pop(p)
                    finish = True
                    p-=1
    else:
        for p in range(0,len(processPool)):
                if p >= len(processPool):
                    break

                result=processPool[p].poll()
                if result is not None:
                    print("result",result)
                    if result!=0:
                        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                        print(processPool[p])
                        print(processPool[p].args)
                        errors.append(' '.join(processPool[p].args))
                        #with open("errors.txt",'a') as f:
                        #    print(' '.join(processPool[p].args),file=f)
                    processPool.pop(p)
                    finish = True
                    p-=1

    try:
        processPool.append(subprocess.Popen(cmd))
    except:
        print(len(processPool))

    #data=remove_first_line('./errors.txt', './errors.txt')

if errors:
    with open("erros.txt",'w') as f:
        [print(i) for i in errors]
