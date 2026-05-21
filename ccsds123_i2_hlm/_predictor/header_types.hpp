#pragma once
#include <cstdint>

enum class SampleType : uint8_t
{
  UNSIGNED_INTEGER = 0,
  SIGNED_INTEGER = 1
};

enum class LargeDFlag : uint8_t
{
  SMALL_D = 0,
  LARGE_D = 1
};

enum class SampleEncodingOrder : uint8_t
{
  BI = 0,
  BSQ = 1
};

enum class EntropyCoderType : uint8_t
{
  SAMPLE_ADAPTIVE = 0,
  HYBRID = 1,
  BLOCK_ADAPTIVE = 2
};

enum class QuantizerFidelityControlMethod : uint8_t
{
  LOSSLESS = 0,
  ABSOLUTE_ONLY = 1,
  RELATIVE_ONLY = 2,
  ABSOLUTE_AND_RELATIVE = 3
};

enum class TableType : uint8_t
{
  UNSIGNED_INTEGER = 0,
  SIGNED_INTEGER = 1,
  FLOAT = 2
};

enum class TableStructure : uint8_t
{
  ZERO_DIMENSIONAL = 0,
  ONE_DIMENSIONAL = 1,
  TWO_DIMENSIONAL_ZX = 2,
  TWO_DIMENSIONAL_YX = 3
};

enum class SampleRepresentativeFlag : uint8_t
{
  NOT_INCLUDED = 0,
  INCLUDED = 1
};

enum class PredictionMode : uint8_t
{
  FULL = 0,
  REDUCED = 1
};

enum class WeightExponentOffsetFlag : uint8_t
{
  ALL_ZERO = 0,
  NOT_ALL_ZERO = 1
};

enum class LocalSumType : uint8_t
{
  WIDE_NEIGHBOR_ORIENTED = 0,
  NARROW_NEIGHBOR_ORIENTED = 1,
  WIDE_COLUMN_ORIENTED = 2,
  NARROW_COLUMN_ORIENTED = 3
};

enum class WeightExponentOffsetTableFlag : uint8_t
{
  NOT_INCLUDED = 0,
  INCLUDED = 1
};

enum class WeightInitMethod : uint8_t
{
  DEFAULT = 0,
  CUSTOM = 1
};

enum class WeightInitTableFlag : uint8_t
{
  NOT_INCLUDED = 0,
  INCLUDED = 1
};

enum class PeriodicErrorUpdatingFlag : uint8_t
{
  NOT_USED = 0,
  USED = 1
};

enum class ErrorLimitAssignmentMethod : uint8_t
{
  BAND_INDEPENDENT = 0,
  BAND_DEPENDENT = 1
};

enum class BandVaryingDampingFlag : uint8_t
{
  BAND_INDEPENDENT = 0,
  BAND_DEPENDENT = 1
};

enum class DampingTableFlag : uint8_t
{
  NOT_INCLUDED = 0,
  INCLUDED = 1
};

enum class BandVaryingOffsetFlag : uint8_t
{
  BAND_INDEPENDENT = 0,
  BAND_DEPENDENT = 1
};

enum class OffsetTableFlag : uint8_t
{
  NOT_INCLUDED = 0,
  INCLUDED = 1
};

enum class AccumulatorInitTableFlag : uint8_t
{
  NOT_INCLUDED = 0,
  INCLUDED = 1
};

enum class RestrictedCodeOptionsFlag : uint8_t
{
  UNRESTRICTED = 0,
  RESTRICTED = 1
};
