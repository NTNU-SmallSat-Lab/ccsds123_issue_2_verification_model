image = raw_images/Landsat_mountain-u16be-6x50x100.raw
image_format = u16be
hex_length = 320
bin_length = 84

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
	@echo "\n\nheader:"; \
	xxd -b -c 1 test/header.bin | cut -d' ' -f 2 | tr -d '\n'
	@echo "\n\nGolden model:"; \
	xxd -b -c 1 test/golden.bin | cut -d' ' -f 2 | tr -d '\n'
	@echo "\n\nHigh-level model:"; \
	xxd -b -c 1 test/hlm.bin | cut -d' ' -f 2 | tr -d '\n'
	@echo "\n\n"; \


compare:
	make clean; \
	python ccsds123_0_b_2_high_level_model.py $(image); \
	cp output/header.bin test/; \
	cp output/z-output-bitstream.bin test/hlm.bin; \
	cp output/hybrid_initial_accumulator.bin test/hybrid_initial_accumulator.bin; \
	lcnl_bsq_reader output/header.bin $(image_format) $(image) | lcnl_encoder -o output/optional_tables.bin -q output/error_limits.bin -a test/hybrid_initial_accumulator.bin output/header.bin $(image_format) /dev/stdin test/golden.bin; \
	python tools/files_identical_check.py test/golden.bin test/hlm.bin; \
	make print > test/comparison.txt

# Example: make compare_with_header image=Test1-20190201/sample-000000-s32be-33x1x2.raw header=Test1-20190201/sample-000000-hdr.bin image_format=s32be
compare_with_header:
	make clean; \
	python ccsds123_0_b_2_high_level_model.py $(image) --header $(header); \
	cp output/header.bin test/; \
	cp output/z-output-bitstream.bin test/hlm.bin; \
	lcnl_encoder $(header) $(image_format) $(image) test/golden.bin
	@echo "Header: "; \
	python tools/files_identical_check.py test/header.bin $(header)
	@echo "\nCompressed image: "; \
	python tools/files_identical_check.py test/golden.bin test/hlm.bin; \
	make print > test/comparison.txt

# Example: make compare_with_header_optional_tables optional_tables=optional_tables.bin header=header.bin error_limits=error_limits.bin
compare_with_header_optional_tables:
	make clean; \
	python ccsds123_0_b_2_high_level_model.py $(image) --header $(header) --optional $(optional_tables) --error_limits $(error_limits); \
	cp output/header.bin test/; \
	cp output/z-output-bitstream.bin test/hlm.bin; \
	cp output/hybrid_initial_accumulator.bin test/hybrid_initial_accumulator.bin; \
	lcnl_bsq_reader output/header.bin $(image_format) $(image) | lcnl_encoder -o $(optional_tables) -q $(error_limits) -a test/hybrid_initial_accumulator.bin output/header.bin $(image_format) /dev/stdin test/golden.bin; \
	# lcnl_encoder -q $(error_limits) -o $(optional_tables) $(header) $(image_format) $(image) test/golden.bin
	@echo "Header: "; \
	python tools/files_identical_check.py test/header.bin $(header)
	@echo "\nCompressed image: "; \
	python tools/files_identical_check.py test/golden.bin test/hlm.bin; \
	make print > test/comparison.txt

test_reading_tables:
	make compare; \
	cp output/header.bin .; \
	cp output/optional_tables.bin .; \
	cp output/error_limits.bin .; \
	make compare_with_header_optional_tables optional_tables=optional_tables.bin header=header.bin error_limits=error_limits.bin

# Example: make compare_vector image=../Test1-20190201/sample-000001-u32be-4x11x45.raw header=../Test1-20190201/sample-000001-hdr.bin image_format=s32be correct=../Test1-20190201/sample-000001.flex
compare_vector:
	make clean; \
	python ccsds123_0_b_2_high_level_model.py $(image) --header $(header) --optional $(optional_tables) --error_limits $(error_limits) --accu $(accu); \
	cp output/header.bin test/; \
	cp output/z-output-bitstream.bin test/hlm.bin; \
	cp $(correct) test/golden.bin
	@echo "Header: "; \
	python tools/files_identical_check.py test/header.bin $(header)
	@echo "\nhlm and correct image: "; \
	python tools/files_identical_check.py test/golden.bin test/hlm.bin; \
	make print > test/comparison.txt

clean:
	rm -f test/*
	rm -f output/*

push_to_server:
	scp verify.py $(server):$(folder)
	scp ccsds123_0_b_2_high_level_model.py $(server):$(folder)
	scp Makefile $(server):$(folder)
	scp -r src $(server):$(folder)
