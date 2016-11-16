all:
	(cd release;	zip ../release.zip *)

clean:
	rm -f *.hlt *.log *.zip debug/*
