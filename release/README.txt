running the file:

	python3 main.py --filename [filename] {--filters [filters] -p --plot}
	
flags in {} are optional.

analyses all of the signals in a file and determines which are faulty. analysis is done with three different filters which find segments in the signal where a value keeps repeating (uniq), the signal does not significantly change (segment) and where the signal contains abrupt spikes (gradient). which of these filters are used is togglable. if needed, the program can also test the physicality of the signals, though this makes the program run much longer and the result may be unreliable if the dataset contains many bad or unphysical signals. 

flags:
	--filename: path to the file to containing the data to be analysed
	--filters: a list of the filters to be used. accepted filters are "uniq", "segment" and 
		  "gradient". all filters are used if no filters are specified.
	-p: the program analyses the physicality of all signals if this flag is used. takes 
	    significantly longer than the filters.
	--plot: plots all of the signals with all of the suspicious and bad segments found by
	        the filters as well as whether or not the signal was determined to be bad and/or
	        physical
