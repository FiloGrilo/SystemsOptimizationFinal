# SystemsOptimizationFinal
Final result of the group project for the course 02229 Systems Optimization

To run the program, modify the initial variables "parent_dir", "testcases_path" and "test_file".

The "parent_dir" should be the parent directory to the folder with the testcases folders. The folder with the execution logs will also be located here.

The "testcases_path" should be the name of the folder placed in the "parent_dir" path that contains the .csv testcase files.

The "test_file" should be the name of the specific testcase file to be ran.

As an example, to test the file with the following path "/Users/joaomena/Documents/testcases_seperation_tested/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__15__tsk.csv", the variables should be set like this:

> parent_dir = "/Users/joaomena/Documents/"

> testcases_path = os.path.join(parent_dir, "testcases_seperation_tested")

> test_file = "taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__15__tsk.csv"

A virtual environment should be created using the command:

> python3 -m venv venv  

The required dependencies can be installed using the command:

> pip3 install -r requirements.txt

To activate the virtual environment run the following command:

> source venv/bin/activate

After following the mentioned steps, you should be ready to run the program with the command:

> python3 main.py
