Project consists of 2 main folders, cfrBasic and training.

cfrWhist.sln is a visual studio project, for the generation of CFR data of a simplified version of whist, with both single and multithread support.
It is based on an implementation found here, https://github.com/brianberns/Cfrm.

In program.cs, the Main method contains the following values that can be adjusted:
	    bool multiThread = true; //toggle for multi-threading
            bool fileMergingEnabled = false; //toggle for merging of strategy files
            var numIterations = 100000; //number of iterations to run
            int progressInterval = numIterations/10; //interval to print progress


S