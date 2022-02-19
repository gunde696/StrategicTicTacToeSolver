using SIMD

isXO = UInt64(9223372036854775808)

# A board is represented as a tuple of (UInt64, UInt64, UInt64).
# The first value stores 64 bits. In order: 27 for the first row of X boards, 27 for the first row of O boards, 9 for the big X board, and 1 bit for whose turn it is
# The second value stores 63 bits of information. In order: 27 for the second row of X boards, 27 for the second row of O boards, and 9 for the big O board.
# The third value stores 63 bits as well! In order: 27 for the third row of X boards, 27 for the third row of O boards, and 9 for the big CAT board.
# Only 2 bits are wasted!

# a vector containing the bitmask representation of all possible win conditions
wincons = UInt16[UInt16(7), UInt16(56), UInt16(448), UInt16(73), UInt16(146), UInt16(292), UInt16(273), UInt16(84)]
# the SIMD version of the above vector
vWincons = vload(Vec{8, UInt16}, wincons, 1)
# a vector filled with 0, to compare against the resulting SIMD vector
emptyBools = Bool[0, 0, 0, 0, 0, 0, 0, 0]
# given a normal sized tic tac toe board, determines if it is won yet using SIMD operations
# IE, it can work on the overall big board, or on any one small board
# Only determines if that one mask of the board is won
# IE, only determines if X has won, not if O has won if pass in the x bitmask
# returns a single bool
function isWonSIMD(normalBoard::UInt16)::Bool
    # fills a vector with the passed in board bitmask
    normBoardVec = fill(normalBoard, 8)
    # creates a result vector that stores bools
    result = Array{Bool}(undef, 8)
    # creates a range of values to be vectorized later on; vectorizes 8 values starting at the 0 index
    lane = VecRange{8}(0)
    # loads the passed in board to a SIMD vector
    vBigBoard = vload(Vec{8, UInt16}, normBoardVec, 1)
    # stores the desired value in the result vector
    # for each possible wincon, it sees if the passed in mask contains that wincon
    # this is done via bitwise and and the == operation
    result[lane + 1] = (vBigBoard & vWincons == vWincons)
    # if the board contains no wincons, and is then full of 0, it will equal 'emptyBools', and then return false. Otherwise, it returns true
    return result != emptyBools
end

# returns the bits in a certain window from [startIndex, endIndex) in the form of a UInt16
function subBits16(value::UInt64, startIndex, endIndex)::UInt16
    return UInt16((value % (1 << endIndex)) >> startIndex)
end

# returns the bits in a certain window from [startIndex, endIndex) in the form of a UInt128
function subBits128(value::UInt64, startIndex, endIndex)::UInt128
    return UInt128((value % (1 << endIndex)) >> startIndex)
end

function emptySpaces(board::Tuple{UInt64, UInt64, UInt64})::UInt128
    # don't worry about it. It's too looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong.
    # but in all actuality, it just logical 'ands' all of the bit maps of the whole board for x, o, and cats at the correct locations
    # if a bit is turned on, that spot, or board, is full, or won. otherwise, it is empty or unwon.
    # the first 81 bits (0 - 81) store the spots of the board, while the next 9 (81 - 90) store the big boards.
    return (subBits128(board[1], 0, 27) & subBits128(board[1], 27, 54)) | ((subBits128(board[2], 0, 27) & subBits128(board[2], 27, 54)) << 27) | ((subBits128(board[3], 0, 27) & subBits128(board[3], 27, 54)) << 54) | ((subBits128(board[1], 54, 63) & subBits128(board[2], 54, 63) & subBits128(board[3], 54, 63)) << 81)
end

# the bitmasks corresponding to the locations of every spot on the board.
locations = (UInt64(1), UInt64(2), UInt64(4), UInt64(8), UInt64(16), UInt64(32), UInt64(64), UInt64(128), UInt64(256), UInt64(512), UInt64(1024), UInt64(2048), UInt64(4096), UInt64(8192), UInt64(16384), UInt64(32768), UInt64(65536), UInt64(131072), UInt64(262144), UInt64(524288), UInt64(1048576), UInt64(2097152), UInt64(4194304), UInt64(8388608), UInt64(16777216), UInt64(33554432), UInt64(67108864), UInt64(134217728), UInt64(268435456), UInt64(536870912), UInt64(1073741824), UInt64(2147483648), UInt64(4294967296), UInt64(8589934592), UInt64(17179869184), UInt64(34359738368), UInt64(68719476736), UInt64(137438953472), UInt64(274877906944), UInt64(549755813888), UInt64(1099511627776), UInt64(2199023255552), UInt64(4398046511104), UInt64(8796093022208), UInt64(17592186044416), UInt64(35184372088832), UInt64(70368744177664), UInt64(140737488355328), UInt64(281474976710656), UInt64(562949953421312), UInt64(1125899906842624), UInt64(2251799813685248), UInt64(4503599627370496), UInt64(9007199254740992))
# play() takes in a board and a location, and returns a new board (tuples are immutable)
function play(board::Tuple{UInt64, UInt64, UInt64}, location)::Tuple{UInt64, UInt64, UInt64}
    # the if - ifelse - else block determines which row is being played on
    if location < 27
        # first, we need to check if a small board will be won by this play
        # boardWasWon = isBoardWon((board[1] | locations[location + 1 + (board[1] & isXO == isXO ? 27 : 0)], board[2], board[3]), location)

        # second, check if the board was cat
        #= if !boardWasWon
               boardWasCat = (subBits16(board[1] | locations[location + 1 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9, UInt64(location ÷ 9)*9 + 9) | subBits16(board[1] | locations[location + 1 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9 + 27, UInt64(location ÷ 9)*9 + 36)) == 511
        end =#

        # (1 << UInt64(location ÷ 9))

        # return the board with the correct move played, and possibly a modified won or cat board.
        if isBoardWon((board[1] | locations[location + 1 + (board[1] & isXO == isXO ? 27 : 0)], board[2], board[3]), location)
            # checks if the game is completely over.
            # uses the high-speed SIMD method to determine if the current players big board contains a winning combo
            if isWonSIMD(UInt16(board[1] & isXO != isXO ? (subBits16(board[1], 54, 63) | (1 << UInt64(location ÷ 9))) : (subBits16(board[2], 54, 63) | (1 << UInt64(location ÷ 9)))))
                # If a board has been won, returns the maximum value in one spot, and zero in the others.
                # For X, it returns: (max, 0, 0). For O, it returns: (0, max, 0). For CAT, it returns (0, 0, max)
                return board[1] & isXO != isXO ? (UInt64(18446744073709551615), UInt64(0), UInt64(0)) : (UInt64(0), UInt64(18446744073709551615), UInt64(0))
            # Checks if the game is CAT by seeing if the 'or' of all three board masks is full.
            elseif (subBits16(board[1], 54, 63) | subBits16(board[2], 54, 63) | subBits16(board[3], 54, 63) | (1 << UInt64(location ÷ 9))) == 511
                return (UInt64(0), UInt64(0), UInt64(18446744073709551615))

            # if not, checks if x or o was the player that won the board
            elseif board[1] & isXO != isXO
                # or's the bitmask of the won board with the first or second board value,
                # which stores x's and o's won boards, respectively.
                return (((board[1] | locations[location + 1 + (board[1] & isXO == isXO ? 27 : 0)]) ⊻ isXO) | (1 << (54 + UInt64(location ÷ 9))), board[2], board[3])
            # this is for if o won the board
            else
                return (((board[1] | locations[location + 1 + (board[1] & isXO == isXO ? 27 : 0)]) ⊻ isXO), board[2] | (1 << (54 + UInt64(location ÷ 9))), board[3])
            end
        # If the board was cat, must store that in the third value of board.
        elseif (subBits16(board[1] | locations[location + 1 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9, UInt64(location ÷ 9)*9 + 9) | subBits16(board[1] | locations[location + 1 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9 + 27, UInt64(location ÷ 9)*9 + 36)) == 511
            # also must check if the whole game is cat due to the board being cat-ed
            if (subBits16(board[1], 54, 63) | subBits16(board[2], 54, 63) | subBits16(board[3], 54, 63) | (1 << UInt64(location ÷ 9))) == 511
                return (UInt64(0), UInt64(0), UInt64(18446744073709551615))
            # if not, then return the board as expected with modifications
            else
                return ((board[1] | locations[location + 1 + (board[1] & isXO == isXO ? 27 : 0)]) ⊻ isXO, board[2], board[3] | (1 << (54 + UInt64(location ÷ 9))))
            end
        else
            # board[1] is the only value being modified, as that value contains the spot that is being played on
            # that value is being bitwise | to the correct location value in locations
            # that value is found by taking the location at (location + 1) since indexing starts at 1
            # then (board[1] & isXO == isXO ? 27 : 0) must be added to the index to account for the stagger to o's indices.
            # board[1] is then always xor-ed (which is this symbol: ⊻) with isXO, as isXO is the bit that stores if it is x's or o's turn
            return ((board[1] | locations[location + 1 + (board[1] & isXO == isXO ? 27 : 0)]) ⊻ isXO, board[2], board[3])
        end
    # Do the same for the second row of boards
    elseif location < 54
        #boardWasWon = isBoardWon((board[1], board[2] | locations[location + 1 - 27 + (board[1] & isXO == isXO ? 27 : 0)], board[3]), location)

        #=if !boardWasWon
            boardWasCat = (subBits16(board[2] | locations[location + 1 - 27 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9 - 27, UInt64(location ÷ 9)*9 - 18) | subBits16(board[2] | locations[location + 1 - 27 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9, UInt64(location ÷ 9)*9 + 9)) == 511
        end=#

        if isBoardWon((board[1], board[2] | locations[location + 1 - 27 + (board[1] & isXO == isXO ? 27 : 0)], board[3]), location)
            if isWonSIMD(UInt16(board[1] & isXO != isXO ? (subBits16(board[1], 54, 63) | (1 << UInt64(location ÷ 9))) : (subBits16(board[2], 54, 63) | (1 << UInt64(location ÷ 9)))))
                return board[1] & isXO != isXO ? (UInt64(18446744073709551615), UInt64(0), UInt64(0)) : (UInt64(0), UInt64(18446744073709551615), UInt64(0))
            elseif (subBits16(board[1], 54, 63) | subBits16(board[2], 54, 63) | subBits16(board[3], 54, 63) | (1 << UInt64(location ÷ 9))) == 511
                return (UInt64(0), UInt64(0), UInt64(18446744073709551615))
            elseif board[1] & isXO != isXO
                return ((board[1] ⊻ isXO) | (1 << (54 + UInt64(location ÷ 9))), board[2] | locations[location + 1 - 27 + (board[1] & isXO == isXO ? 27 : 0)], board[3])
            else
                return ((board[1] ⊻ isXO), board[2] | (1 << (54 + UInt64(location ÷ 9))) | locations[location + 1 - 27 + (board[1] & isXO == isXO ? 27 : 0)], board[3])
            end
        elseif (subBits16(board[2] | locations[location + 1 - 27 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9 - 27, UInt64(location ÷ 9)*9 - 18) | subBits16(board[2] | locations[location + 1 - 27 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9, UInt64(location ÷ 9)*9 + 9)) == 511
            if (subBits16(board[1], 54, 63) | subBits16(board[2], 54, 63) | subBits16(board[3], 54, 63) | (1 << UInt64(location ÷ 9))) == 511
                return (UInt64(0), UInt64(0), UInt64(18446744073709551615))
            else
                return (board[1] ⊻ isXO, board[2] | locations[location + 1 - 27 + (board[1] & isXO == isXO ? 27 : 0)], board[3] | (1 << (54 + UInt64(location ÷ 9))))
            end
        else
            return (board[1] ⊻ isXO, board[2] | locations[location + 1 - 27 + (board[1] & isXO == isXO ? 27 : 0)], board[3])
        end
    # And the third row of boards
    else
        #boardWasWon = isBoardWon((board[1], board[2], board[3] | locations[location + 1 - 54 + (board[1] & isXO == isXO ? 27 : 0)]), location)

        #=if !boardWasWon
            boardWasCat = (subBits16(board[3] | locations[location + 1 - 54 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9 - 54, UInt64(location ÷ 9)*9 - 45) | subBits16(board[3] | locations[location + 1 - 54 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9 - 27, UInt64(location ÷ 9)*9 - 18)) == 511
        end=#

        if isBoardWon((board[1], board[2], board[3] | locations[location + 1 - 54 + (board[1] & isXO == isXO ? 27 : 0)]), location)
            if isWonSIMD(UInt16(board[1] & isXO != isXO ? (subBits16(board[1], 54, 63) | (1 << UInt64(location ÷ 9))) : (subBits16(board[2], 54, 63) | (1 << UInt64(location ÷ 9)))))
                return board[1] & isXO != isXO ? (UInt64(18446744073709551615), UInt64(0), UInt64(0)) : (UInt64(0), UInt64(18446744073709551615), UInt64(0))
            elseif (subBits16(board[1], 54, 63) | subBits16(board[2], 54, 63) | subBits16(board[3], 54, 63) | (1 << UInt64(location ÷ 9))) == 511
                return (UInt64(0), UInt64(0), UInt64(18446744073709551615))
            elseif board[1] & isXO != isXO
                return ((board[1] ⊻ isXO) | (1 << (54 + UInt64(location ÷ 9))), board[2], board[3] | locations[location + 1 - 54 + (board[1] & isXO == isXO ? 27 : 0)])
            else
                return ((board[1] ⊻ isXO), board[2] | (1 << (54 + UInt64(location ÷ 9))), board[3] | locations[location + 1 - 54 + (board[1] & isXO == isXO ? 27 : 0)])
            end
        elseif (subBits16(board[3] | locations[location + 1 - 54 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9 - 54, UInt64(location ÷ 9)*9 - 45) | subBits16(board[3] | locations[location + 1 - 54 + (board[1] & isXO == isXO ? 27 : 0)], UInt64(location ÷ 9)*9 - 27, UInt64(location ÷ 9)*9 - 18)) == 511
            if (subBits16(board[1], 54, 63) | subBits16(board[2], 54, 63) | subBits16(board[3], 54, 63) | (1 << UInt64(location ÷ 9))) == 511
                return (UInt64(0), UInt64(0), UInt64(18446744073709551615))
            else
                return (board[1] ⊻ isXO, board[2], board[3] | locations[location + 1 - 54 + (board[1] & isXO == isXO ? 27 : 0)] | (1 << (54 + UInt64(location ÷ 9))))
            end
        else
            return (board[1] ⊻ isXO, board[2], board[3] | locations[location + 1 - 54 + (board[1] & isXO == isXO ? 27 : 0)])
        end
    end
end

# Play many moves in a row given a board and a tuple of locations. Returns the final board state after all the moves
function playMany(board::Tuple{UInt64, UInt64, UInt64}, locations)::Tuple{UInt64, UInt64, UInt64}
    # Loops through every location to be played at
    for location in locations
        # Plays at that location, stores it in board again
        board = play(board, location)
    end
    # Returns the final board
    return board
end

# the bitmasks corresponding to the locations of the boards, IE, the first board (Top-Left), the second board (Top-Middle), and so on
bigBoardMasks = (UInt64(18014398509481984), UInt64(36028797018963968), UInt64(72057594037927936), UInt64(144115188075855872), UInt64(288230376151711740), UInt64(576460752303423490), UInt64(1152921504606846980), UInt64(2305843009213694000), UInt64(4611686018427387900))
# Checks if x or o has won the board they are playing on
# simply input the location of the move; not the bitmask version. IE, the bottom right move is 80, not 2^80.
function isBoardWon(board::Tuple{UInt64, UInt64, UInt64}, location)::Bool
    # checks if it is x's turn
    if board[1] & isXO != isXO
        # sorts the move by which board it is being played on
        # seems dumb, but it works, so it's not dumb
        if location < 9
            # uses the high-speed SIMD method to see if that little board is won
            if isWonSIMD(subBits16(board[1], 0, 9))
                # returns true
                return true
            end
            # repeat for every little board
        elseif location < 18
            if isWonSIMD(subBits16(board[1], 9, 18))
                return true
            end
            # yes, all of them
        elseif location < 27
            if isWonSIMD(subBits16(board[1], 18, 27))
                return true
            end
            # this one too
        elseif location < 36
            if isWonSIMD(subBits16(board[2], 0, 9))
                return true
            end
            # and this one
        elseif location < 45
            if isWonSIMD(subBits16(board[2], 9, 18))
                return true
            end
            # this is the last one I promise
        elseif location < 54
            if isWonSIMD(subBits16(board[2], 18, 27))
                return true
            end
            # I tricked you
        elseif location < 63
            if isWonSIMD(subBits16(board[3], 0, 9))
                return true
            end
            # why are you still here?
        elseif location < 72
            if isWonSIMD(subBits16(board[3], 9, 18))
                return true
            end
            # last one! For now...
        else
            if isWonSIMD(subBits16(board[3], 18, 27))
                return true
            end
        end
    # otherwise, it is o's turn
    else
        # same as for x's, but for o's. True will still be returned, where the method is called has to store it properly
        if location < 9
            # all subBits will be shifted by 27, to access o's locations
            if isWonSIMD(subBits16(board[1], 27, 36))
                return true
            end
            # we've got to check all of these one's too, right?
        elseif location < 18
            if isWonSIMD(subBits16(board[1], 36, 45))
                return true
            end
            # right.
        elseif location < 27
            if isWonSIMD(subBits16(board[1], 45, 54))
                return true
            end
            # you know, I used to think my life was a tragedy
        elseif location < 36
            if isWonSIMD(subBits16(board[2], 27, 36))
                return true
            end
            # now I realized
        elseif location < 45
            if isWonSIMD(subBits16(board[2], 36, 45))
                return true
            end
            # it's a comedy
        elseif location < 54
            if isWonSIMD(subBits16(board[2], 45, 54))
                return true
            end
            # no more Joker quotes for now
        elseif location < 63
            if isWonSIMD(subBits16(board[3], 27, 36))
                return true
            end
            # maybe Bane quotes?
        elseif location < 72
            if isWonSIMD(subBits16(board[3], 36, 45))
                return true
            end
            # and we're done! That wasn't too bad.
        else
            if isWonSIMD(subBits16(board[3], 45, 54))
                return true
            end
        end
    end
    return false
end

# determines if the game has ended. Returns a Tuple(Bool, Bool), where:
# (True, True) -> x has won, (True, False) -> o has won, (False, True) -> cat has won, (False, False) -> game is ongoing
function isEnded(board::Tuple{UInt64, UInt64, UInt64})::Tuple{Bool, Bool}
    # uses the subBits method to find the bitmasks for the big x, o, and cat boards
    xBoard = subBits16(board[1], 54, 63)
    oBoard = subBits16(board[2], 54, 63)
    catBoard = subBits16(board[3], 54, 63)
    # using the isWonSIMD method, determines if x or o have won
    if isWonSIMD(xBoard)
        return (true, true)
    elseif isWonSIMD(oBoard)
        return (true, false)
    # if the logical and of the x, o, and cat boards are a full bitmask (511 = 2^9 - 1), the game is cat
    elseif (xBoard | oBoard | catBoard == 511)
        return (false, true)
    else
        return (false, false)
    end
end

orderOfXSpots = (UInt64(1), UInt64(2), UInt64(4), UInt64(512), UInt64(1024), UInt64(2048), UInt64(262144), UInt64(524288), UInt64(1048576), UInt64(8), UInt64(16), UInt64(32), UInt64(4096), UInt64(8192), UInt64(16384), UInt64(2097152), UInt64(4194304), UInt64(8388608), UInt64(64), UInt64(128), UInt64(256), UInt64(32768), UInt64(65536), UInt64(131072), UInt64(16777216), UInt64(33554432), UInt64(67108864), )
orderOfOSpots = (UInt64(134217728), UInt64(268435456), UInt64(536870912), UInt64(68719476736), UInt64(137438953472), UInt64(274877906944), UInt64(35184372088832), UInt64(70368744177664), UInt64(140737488355328), UInt64(1073741824), UInt64(2147483648), UInt64(4294967296), UInt64(549755813888), UInt64(1099511627776), UInt64(2199023255552), UInt64(281474976710656), UInt64(562949953421312), UInt64(1125899906842624), UInt64(8589934592), UInt64(17179869184), UInt64(34359738368), UInt64(4398046511104), UInt64(8796093022208), UInt64(17592186044416), UInt64(2251799813685248), UInt64(4503599627370496), UInt64(9007199254740992))
bitStagger = UInt64(2^27)
# prints the board. Understanding this I will leave as an exercise to the reader.
function printBoard(board::Tuple{UInt64, UInt64, UInt64})
    println("The Whole Board")
    for row in [1:3;]
        for spotIndex in [1:27;]
            if board[row] & orderOfXSpots[spotIndex] == orderOfXSpots[spotIndex]
                print(" x ")
            elseif (board[row] & orderOfOSpots[spotIndex]) == orderOfOSpots[spotIndex]
                print(" o ")
            else
                print("   ")
            end
            if spotIndex == 9 || spotIndex == 18 || spotIndex == 27
                println()
            end
            if spotIndex == 3 || spotIndex == 6 || spotIndex == 12 || spotIndex == 15 || spotIndex == 21 || spotIndex == 24
                print("|")
            end
        end
        if row != 3
            println("-----------------------------")
        end
    end
    println("Big Board:")
    xBoard = subBits16(board[1], 54, 63)
    println("Bitstring of X: $(bitstring(xBoard))")
    oBoard = subBits16(board[2], 54, 63)
    println("Bitstring of O: $(bitstring(oBoard))")
    catBoard = subBits16(board[3], 54, 63)
    println("Bitstring of CAT: $(bitstring(catBoard))")
    for x in [0:8;]
        boardLocation = 1 << x
        if (xBoard & boardLocation == boardLocation)
            print(" x")
        elseif (oBoard & boardLocation == boardLocation)
            print(" o")
        elseif (catBoard & boardLocation == boardLocation)
            print(" c")
        else
            print("  ")
        end
        if x % 3 == 2
            println()
            if x < 8
                println("-----------")
            end
        else
            print(" |")
        end
    end
end

print("Begin:")
@time board = (UInt64(0), UInt64(0), UInt64(0))

# This set of moves tests to see if winning and catting boards gets registered correctly
#=board = play(board, 0)
board = play(board, 1)
board = play(board, 2)
board = play(board, 3)
board = play(board, 5)
board = play(board, 4)
board = play(board, 6)
board = play(board, 8)
board = play(board, 7)

board = play(board, 45)
board = play(board, 54)
board = play(board, 46)
board = play(board, 55)
board = play(board, 47)
board = play(board, 56)

board = play(board, 9)
board = play(board, 10)
board = play(board, 11)
board = play(board, 12)
board = play(board, 14)
board = play(board, 13)
board = play(board, 15)
board = play(board, 17)
board = play(board, 16)

board = play(board, 72)
board = play(board, 73)
board = play(board, 74)
board = play(board, 75)
board = play(board, 77)
board = play(board, 76)
board = play(board, 78)
board = play(board, 80)
board = play(board, 79)=#

# This set of moves tests to see if random moves get registered correctly
#=board = play(board, 25)
board = play(board, 0)
board = play(board, 80)
board = play(board, 1)
board = play(board, 79)
board = play(board, 2)
board = play(board, 78)
board = play(board, 9)
board = play(board, 77)
board = play(board, 10)
board = play(board, 76)
board = play(board, 11)
board = play(board, 75)
board = play(board, 18)
board = play(board, 74)
board = play(board, 19)
board = play(board, 73)
board = play(board, 20)=#

# This set of moves tests to see if winning the game is registered
#=board = play(board, 0)
board = play(board, 27)
board = play(board, 1)
board = play(board, 28)
board = play(board, 2)
board = play(board, 29)

board = play(board, 9)
board = play(board, 71)
board = play(board, 10)
board = play(board, 72)
board = play(board, 11)
board = play(board, 73)

board = play(board, 18)
board = play(board, 53)
board = play(board, 19)
board = play(board, 54)
board = play(board, 20)
board = play(board, 55)=#

# This set of moves tests to see if catting the game gets registered
#=board = play(board, 0)
board = play(board, 9)
board = play(board, 1)
board = play(board, 10)
board = play(board, 2)
board = play(board, 11)

board = play(board, 18)
board = play(board, 27)
board = play(board, 19)
board = play(board, 28)
board = play(board, 20)
board = play(board, 29)

board = play(board, 45)
board = play(board, 36)
board = play(board, 46)
board = play(board, 37)
board = play(board, 47)
board = play(board, 38)

board = play(board, 54)
board = play(board, 72)
board = play(board, 55)
board = play(board, 73)
board = play(board, 56)
board = play(board, 74)

board = play(board, 63)
board = play(board, 66)
board = play(board, 64)
board = play(board, 67)
board = play(board, 65)=#


print("Empty Spaces: ")
@time emptySpaces(board)
print("Is Ended: ")
@time isEnded(board)
print("Play: ")
@time play(board, 27)
print("Is Board Won: ")
@time isBoardWon(board, 1)

printBoard(board)
