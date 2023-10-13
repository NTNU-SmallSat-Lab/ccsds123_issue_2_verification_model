image = Landsat_mountain-u16be-6x50x100.raw
image_format = u16be
hex_length = 320
bin_length = 48

print:
	@echo "Header:"; \
	xxd -u output/header.bin
	@echo "\nGolden model:"; \
	xxd -u -l $(hex_length) test/golden.bin
	@echo "\nHigh-level model:"; \
	xxd -u -l $(hex_length) test/hlm.bin; \
	echo "\n\nHeader:"; \
	xxd -b output/header.bin
	@echo "\nGolden model:"; \
	xxd -b -l $(bin_length) test/golden.bin
	@echo "\nHigh-level model:"; \
	xxd -b -l $(bin_length) test/hlm.bin

compare:
	python ccsds123_0-b-2_high_level_model.py; \
	cp output/header.bin test/; \
	cp output/z-output-bitstream.bin test/hlm.bin; \
	lcnl_bsq_reader output/header.bin $(image_format) raw_images/$(image) | lcnl_encoder output/header.bin $(image_format) /dev/stdin test/golden.bin; \
	python files_identical_check.py test/golden.bin test/hlm.bin; \
	make print > test/comparison.txt

clean:
	rm -f test/*