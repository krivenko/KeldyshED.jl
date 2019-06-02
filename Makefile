.PHONY: test
test:
	JULIA_PROJECT="$(CURDIR)" julia test/runtests.jl
