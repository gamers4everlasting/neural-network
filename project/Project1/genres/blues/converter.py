
import sys
import os

# Store all command line args in genre_dirs
#genre_dirs = sys.argv[1:]



	# echo contents before altering
cwd = os.getcwd()
os.system("ls")

for filename in os.listdir(cwd):
    if filename.endswith('.au'):
    	os.system("sox " + str(filename) + " " + str(filename[:-3]) + ".wav")


# delete .au from current dir
os.system("rm *.au")
# echo contents of current dir
print('After conversion:')
os.system("ls")
print('\n')


