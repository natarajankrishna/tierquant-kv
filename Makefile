.PHONY: test sim breakeven lint
test:
	pytest -q
sim:
	python benchmarks/sim_hitrate.py --out results/sim_hitrate.csv
breakeven:
	python -m tierquant.breakeven --out results/breakeven.csv
