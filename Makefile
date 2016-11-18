all:
	mkdir sweep
	(cd release;	zip ../release.zip *)

clean:
	rm -f *.hlt *.log *.zip debug/*
