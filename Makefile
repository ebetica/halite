all:
	mkdir -p sweep
	sh -c "$$(curl -fsSL https://raw.githubusercontent.com/HaliteChallenge/Halite/master/environment/install.sh)"

package:
	(cd release;	zip ../release.zip *)

clean:
	rm -f *.hlt *.log *.zip debug/*
