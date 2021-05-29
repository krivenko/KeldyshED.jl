.PHONY: test
test:
	JULIA_PROJECT="$(CURDIR)" julia test/runtests.jl
	JULIA_PROJECT="$(CURDIR)" julia -p 4 test/runtests.jl
