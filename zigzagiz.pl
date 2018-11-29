#!/usr/bin/perl
# zigzagiz.pl - draws ascii lines through ascii files for advanced ascii art moves
# Oswald Berthold, 2002
#
# just found this and dumping on playground
# not sure what's going on precisely, probably needs a clean up 

# vars
$dbg = 0;
$linlen = 0;
$c = 0;

# check args
if($#ARGV == -1) {
  print "usage: $0 inputfile\n";
  exit 0;
}

open IPT, "<".$ARGV[0] or die $!;
$/ = "";
@lins = split /\n/, <IPT>;
$lins = $#lins+1;

foreach $lin (@lins) {
  #$tll = length($lin);
  @linch = split //,$lin;
  $tll = $#linch+1;
  $tc = eval "xr$c";
  @$tc = @linch;
  $lins[$c] = \@$tc;
#  print $tll, " lin: ",$lin, "\n";
  if($tll > $linlen) { $linlen = $tll; }
  $c++;
}


print "=" x $linlen,"\n\n";
#print "lins: ", $lins, "\n";
#print "linen: ", $linlen, "\n";


# get coordinates for lines
$bla = &gen_lines();

for($z=0;$z<=$#{$bla};$z++){

  $bref = \@{$bla[$z]};
  $cc=0;
  foreach $blibli (@{@{$bla}[$z]}) {
    if($cc==0) {$x1 = $blibli;}
    if($cc==1) {$y1 = $blibli;}
    if($cc==2) {$x2 = $blibli;}
    if($cc==3) {$y2 = $blibli;}
    $cc++;
  }
#  print "x1: $x1, y1: $y1, x2: $x2, y2: $y2\n";

  if($x1 == $x2) {
    $chr = "|";
  } elsif ($y1 == $y2){
    $chr = "_";
  } elsif ($y1 > $y2 && $x1 < $x2
	   || $y1 < $y2 && $x1 > $x2) {
    $chr = "\\";
  } elsif ($y1 < $y2 && $x1 > $x2
	   || $y1 > $y2 && $x1 < $x2) {
    $chr = "/";
  }

  # loop over file and try to place the characters for the lines
#  $tlin = \@lins;
  for($c=0;$c<=$#lins;$c++){
    $uplim = (($y1 > $y2) ? $y2 : $y1);
    $botlim = (($y1 > $y2) ? $y1 : $y2);
    $leftlim = (($x1 > $x2) ? $x2 : $x1);
    $rightlim = (($x1 > $x2) ? $x1 : $x2);
    if($c >= $uplim && $c <= $botlim && $uplim != $botlim){
#      print "yeah\n";
      @{$lins[$c]}[$leftlim + ($c - $uplim)] = $chr;
    } elsif ($uplim == $botlim && $c == $uplim) {
      for($xxc=0;$xxc<=($rightlim - $leftlim);$xxc++){
	@{$lins[$c]}[$leftlim + $xxc] = $chr;
      }
    }
#    print @{$lins[$c]}[$leftlim + ($c - $uplim)],"\n";
#    $tlin[];
    foreach $sl (@{$lins[$c]}) {
#      print "$sl";
    }
#    print "\n";
  }


#  foreach $lx (@{@{$bla}[$z]}) {
#    print "$lx",",";
#  }
#  print " - $z\n";
}

# debugging
#if($dbg) {
#  print "BLALAAAAAAAAA\n";
  for($c=0;$c<=$#lins;$c++){
    foreach $sl (@{$lins[$c]}) {
      print "$sl";
    }
    print "\n";
  }
#}



sub gen_lines() {
    # $numlines = 20 + int rand 40;
  $numlines = 10 + int rand 2;
  for($x=0;$x<$numlines;$x++){
#    print $x,"\n";
    $x1 = int rand $linlen;
    $y1 = int rand $lins;
    $way = int rand 3;
    # straight line
    if($way==0){
      $x2 = $x1 + (int rand ($linlen - $x1));
      $y2 = $y1;
    # upward line
    } elsif($way==1){
      $x2 = $x1;
      $y2 = $y1 + (int rand ($lins - $y1));
    } elsif($way==2){
      $x2 = $x1 + (int rand ($linlen - $x1));
      $y2 = ((int rand 2) ? $y1 + $x2 : $y1 - $x2);
    }
    @lini = ($x1,$y1,$x2,$y2);
    $tcx = eval "xrt$x";
    @$tcx = @lini;
    $liniref = \@$tcx;
    $lindefs[$x] = $liniref;

#    if($dbg) {
#      foreach $ld (@{$lindefs[$x]}) {
#	print "$ld",",";
#      }
#      print "\n";
#    }
  }
  if($dbg){print "nulöines: ".$numlines,"\n";}
  return \@lindefs;
}
