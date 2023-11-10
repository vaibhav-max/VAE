import numpy as np
import timeit
import numpy as np

# Define file paths
bwt_file = r"C:\Users\vaibh\OneDrive\Desktop\IISc\Sem 3\Data Analytics\Assignment6\chrX_bwt\chrX_last_col.txt"
idx_file = r"C:\Users\vaibh\OneDrive\Desktop\IISc\Sem 3\Data Analytics\Assignment6\chrX_bwt\chrX_map.txt"
reference_file = r"C:\Users\vaibh\OneDrive\Desktop\IISc\Sem 3\Data Analytics\Assignment6\chrX_bwt\chrX.fa"

# Read the BWT data from the file
with open(bwt_file, 'r') as file:
    bwt = file.readlines()

# Read the index data from the file
with open(idx_file, 'r') as file:
    idx = file.readlines()

# Read the reference data from the file, skipping the first line
with open(reference_file, 'r') as file:
    next(file)  # Skip the first line
    reference = file.readlines()

# Initialize arrays for A, C, G, and T
num_elements = len(bwt)
A, C, G, T = (np.zeros(shape=num_elements, dtype=np.int32) for _ in range(4))

# Count and store A, C, G, T from each 100-length block
counts = {'A': np.array([block.count('A') for block in bwt]),
          'C': np.array([block.count('C') for block in bwt]),
          'G': np.array([block.count('G') for block in bwt]),
          'T': np.array([block.count('T') for block in bwt])}

# Generating ranks in milestones
for base in 'ACGT':
    cumulative_counts = np.cumsum(counts[base])
    counts[base] = cumulative_counts

Milestone = counts


def search(read):
    # Initial Band
    band_start = 0
    band_end = Milestone['A'][-1] + Milestone['C'][-1] + Milestone['G'][-1] + Milestone['T'][-1] + 1

    # Searching in reverse for each suffix
    for i in reversed(range(len(read))):        
        # Character C to be searched
        char = read[i]

        # Generating nearest blocks
        start_block, start_index = divmod(band_start, 100)
        end_block, end_index = divmod(band_end, 100)

        # Rank of first and last C
        rank_first = -1
        rank_last = -1
        
        # Searching for the first C in the band
        rank_first = Milestone[char][start_block] - bwt[start_block][start_index:].count(char) + 1

        # Searching for the last C in the band
        rank_last = Milestone[char][end_block] - bwt[end_block].count(char) + bwt[end_block][:end_index+1].count(char)

        if rank_last < rank_first:
            return [band_start - 1, band_end - 1], i + 1

        if char == 'A':
            band_start, band_end = rank_first - 1, rank_last - 1
        elif char == 'C':
            band_start, band_end = Milestone['A'][-1] + rank_first - 1, Milestone['A'][-1] + rank_last - 1
        elif char == 'G':
            band_start, band_end = Milestone['A'][-1] + Milestone['C'][-1] + rank_first - 1, Milestone['A'][-1] + Milestone['C'][-1] + rank_last - 1
        elif char == 'T':
            band_start, band_end = Milestone['A'][-1] + Milestone['C'][-1] + Milestone['G'][-1] + rank_first - 1, Milestone['A'][-1] + Milestone['C'][-1] + Milestone['G'][-1] + rank_last - 1

    return [band_start, band_end], 0


#Extract given string starting from given index from reference string
def extract(index, length):
    # Identify character block and offset
    block, start = divmod(index, 100)

    # Initialize the result string
    result = ""

    # Extract from the first block
    result += reference[block][start:]

    # Extract from consecutive blocks
    while len(result) < length:
        block += 1
        result += reference[block]

    # Trim the extra tail characters
    result = result[:length]

    return result


#Utility function to count mismatches
#Returns True if mismactes <=2
def count_miss(str1, str2):
    if len(str1) != len(str2):
        return False

    mis = sum(s1 != s2 for s1, s2 in zip(str1, str2))
    
    return mis <= 2


#Generate Reverse Complement of a string
def reverseComplement(read: str):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    result = [complement.get(base, base) for base in reversed(read)]
    return ''.join(result)


start = timeit.default_timer()
# Initialize arrays to maintain counts for matches in each exon of Red and Green genes
num_exons = 6
RExons, GExons, Exons = np.zeros(num_exons), np.zeros(num_exons), np.zeros(num_exons)

#Reading reads
with open(r"C:\Users\vaibh\OneDrive\Desktop\IISc\Sem 3\Data Analytics\Assignment6\chrX_bwt\reads", 'r') as file:
    for read in file:
        read = read[:-1].replace('N', 'A')
        readrevcomp = reverseComplement(read)
        R, G = np.zeros(6), np.zeros(6)

        for read_ in [read, readrevcomp]:
            band, shift = search(read_)

            for i in range(band[0], band[1] + 1):
                id = int(idx[i]) - shift
                ref = extract(id, len(read_))

                if count_miss(ref, read_):
                    exon_ranges = [
                        ((149249757, 149249868), (149288166, 149288277)),
                        ((149256127, 149256423), (149293258, 149293554)),
                        ((149258412, 149258580), (149295542, 149295710)),
                        ((149260048, 149260213), (149297178, 149297343)),
                        ((149261768, 149262007), (149298898, 149299137)),
                        ((149264290, 149264400), (149301420, 149301530))
                    ]

                    for j, (red_range, green_range) in enumerate(exon_ranges):
                        if red_range[0] <= id <= red_range[1]:
                            R[j] = 1
                        if green_range[0] <= id <= green_range[1]:
                            G[j] = 1

        for i in range(6):
            Exons[i] += (R[i] + G[i]) / 2
            RExons[i] += R[i]
            GExons[i] += G[i]

#The final result Count is summation of Exons
print(Exons)
stop = timeit.default_timer()
print('Time: ', stop - start)

print(RExons)
print(GExons)