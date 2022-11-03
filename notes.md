C:\Users\etudiant.jeip\Documents\tunable-filter\

Conda environment : penguin


##### FOLDER COMPONENTS


- characterize_filter.ipynb does fig. 1.1

\preisach : 

	\memory_study : 2 python files that make sure that in the slow limit, there is no speed dependence.
		- measure_memory_properties.py : verifies that memory isn't affecting by sweeping, then waiting, the re-sweeping
		- measure_speed_dep.py : studies dependence on sweep speed

	\preisach_verif : figures from other programs get saved here.

	- UNUSED_data_analysis.ipynb : some old copy of the TWPA thing. Can be safely ignored.
	- live_filter_control.ipynb : empty file (corruption ? or just uselsess)
	- live_aim_test.ipynb : creates iPyWidget to control the filter live (load Preisach. You input a freq, Preisach runs, sends the command to the filter, then VNA takes a trace and compares the result to expected result)
	- test_preisach_model.ipynb : Runs filter through a voltage signal, measuring its frequency, then compares the frequency with what the Preisach model predicts from the input voltage.
	- test_transducer_aim_		#From a set of desired freqs as a function of time, aiming algorithm predicts voltages. Feeds the voltages into the filter, measuring its output, and compares it to the desired output.
			    |damped.ipynb : input is a damped sine visiting the whole freq range. Stress test for memory array length.
			    |modulated.ipynb : input is a modulated sine. Stress test for repeating the same cycle over and over (making sur there is no drift)

	- measure_preisach_triangle.py : runs a measurement of the first order transition curves and saves it as a file for use by the Preisach code.
	- plot_preisach_triangle.py : Plots hysteresis in the Preisach triangle.