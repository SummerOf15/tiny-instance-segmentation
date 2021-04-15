"""
This code aims to generate the train.txt and test.txt

"""
import random



def compare_txt():
    all_file=open("ImageSets/Main/all.txt").readlines()
    train_file=open("ImageSets/Main/train.txt").readlines()
    test_file=open("ImageSets/Main/test.txt").readlines()

    total_missing=0
    total_duplicate=0
    for line in all_file:
        line=line.strip(" ")
        if line in train_file and line in test_file:
            print("duplicate item {}".format(line))
            total_duplicate+=1
        elif line not in train_file and line not in test_file:
            print("missing item {}".format(line))
            total_missing+=1
    print("total missing: {}".format(total_missing))
    print("total duplicate: {}".format(total_duplicate))


def generate_txt():
    random.seed(1)

    all_file=open("ImageSets/Main/all.txt","r").readlines()
    test_file=open("ImageSets/test.txt","w")
    train_file=open("ImageSets/train.txt","w")

    test=random.sample(all_file,45)
    train=[x for x in all_file if x not in test]
    test_file.write("".join(test))
    train_file.write("".join(train))

    test_file.close()
    train_file.close()
    

if __name__ == "__main__":
    generate_txt()

