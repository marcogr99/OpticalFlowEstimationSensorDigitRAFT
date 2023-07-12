1- SCARICARE e COMPILARE TUTTI I FILE CON IL COMANDO 'make' :
	- EpicFlow
	- DeepMatching
	- Edgedetector ( Toolmaster e edges-master) --> scaricare il compilatore c++ su matlab
	Please compile mex code from within Matlab (note: win64/linux64 binaries included):
  		mex private/edgesDetectMex.cpp -outdir private [OMPPARAMS]
  		mex private/edgesNmsMex.cpp    -outdir private [OMPPARAMS]
  		mex private/spDetectMex.cpp    -outdir private [OMPPARAMS]
  		mex private/edgeBoxesMex.cpp   -outdir private
	Here [OMPPARAMS] are parameters for OpenMP and are OS and compiler dependent.
  		Windows:  [OMPPARAMS] = '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  		Linux V1: [OMPPARAMS] = '-DUSEOMP' CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
  		Linux V2: [OMPPARAMS] = '-DUSEOMP' CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
		To compile without OpenMP simply omit [OMPPARAMS]; note that code will be single threaded in this case.

d) Add edge detection code to Matlab path (change to current directory first): 
 >> addpath(pwd); savepath;


2- FAR FUNZIONARE TUTTO:

	2.1 Da MATLAB: Calcolare gli edges della prima immagine della coppia di frames
		2.1.1 Prendi il file binario calcolato e portalo in Linux
	
	2.2 Da Terminale: Calcolare il matching con deepmatching 
		(comando da dentro deepmatching folder:' ./deepmatching <frame1> <frame 2> -png_settings -out <nome_matchfile>)
		2.2.1 per visualizzare l'uscita di deepmatching:
			comando: ./deepmatching <frame1> <frame 2> -png_settings -nt 0 | python3 viz.py <frame1> <frame2>
	
	2.3 Da Terminale: Calcolare il flusso ottico con EpicFlow: 
		(EpicFlow directory)./epicflow <frame1> <frame 2> <edgesfile> <matchfile> <nome_flusso_output>.flo

3- VISUALIZZARE IL FLUSSO:

	3.1 Scaricare tutta la cartella su vision.middlebury.edu/flow/code/flow-code/
	3.2 Compilare con make all (make clean --> make all) (se non trova gli header, portarli fuori)
	3.3 Da Terminale: dentro la directory dove si è salvato tutti i file in questione:
		./color_flow <path to file.flo> <nome output con ext(.png)>
