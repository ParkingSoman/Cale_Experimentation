# Layer 3, small layer K=10 (local_rand): units to describe

For each unit: activations at both poles on two text sets (the token at [[ ]] is where it fires),
what it writes (direct effect on next-word scores), and which tokens get worse when it is zeroed.
Units named T* are the trained small-layer units. Units named O* are original-layer units that carry
part of the same signal.


(T2: parent pole +, 13 coverage children, children explain 100% of its top tokens)

## T2

Positive pole, set A:
  - +10.91  a generous brother , a faithful husband , a loyal friend , a dutiful official , and a patriotic subject . "    Below is an example of one of Du Fu 's later works , To My Retired Friend Wei ( Chinese : �[[�]]
  - +9.82  @,@ 000 ft ) at an elevation of 85 degrees . The maximum effective rate of fire was only between 10 and 120 rounds per minute because of the frequent need to change the fifteen @[[-]]
  - +9.61  his aunt . He had an elder brother , who died young . He also had three half brothers and one half sister , to whom he frequently refers in his poems , although he never mentions his stepmother .    The son of a scholar @[[-]]
  - +9.52 1931 – 33 the AA guns were replaced with eight 40 @-@ caliber 12 @.@ 7 cm ( 5 @.@ 0 in ) Type 89 dual @[[-]]
  - +9.25  " translate out " the references in those works which he does select .    Other translators have placed much greater weight on trying to convey a sense of the poetic forms used by Du Fu . Vikram Seth in Three Chinese Poets uses English @[[-]]
  - +9.24  be lowered . The recoil mechanism of the guns was also changed from a hydraulic to a pneumatic system , which allowed for a faster firing cycle of the main guns .    By World War II , the guns used Type 91 armour @[[-]]
  - +9.23  even keel after three @-@ days pumping . Hyūga was a primary focus of attack and she received 10 direct hits and up to 30 near misses . She was badly damaged with some 200 @[[-]]
  - +9.16 28 – 29 . A catapult and a collapsible 4 @- tonne ( 3 @.@ 9 @-@ long @-@ ton ) crane were fitted on the stern during the mid @[[-]]
Positive pole, set B:
  - +10.46  by King who said that he hates the track : " That 's one of the black marks in our history , in my book . I just fucking think it 's horrible . [ Laughs ] I hate the opening r[[iff]]
  - +9.54  Recording Industry Association of America .    In order to offset the pace of the group 's previous album , Slayer deliberately slowed down the album 's tempo . In contrast to their previous , the band utilized undistorted guitars and toned @[[-]]
  - +9.35  firm definition for these terms . Among these lesser deities , Egyptologist Claude Traunecker draws a distinction between " genies " — specialized patron spirits of certain places , objects , or activities , such as the sea or marsh god Wadj @[[-]]
  - +9.30  That 's one of the black marks in our history , in my book . I just fucking think it 's horrible . [ Laughs ] I hate the opening riff . It 's what we call a ' happy r[[iff]]
  - +9.15  at the Spring 2004 Jägermeister Music Tour . A live rendition of " South of Heaven " was also included on a bonus DVD which came with the group 's 2007 re @[[-]]
Negative pole, set A:
  - -3.11  and has been certified gold by the Recording Industry Association of America ( RIAA ) for shipments of 500 @,@ 000 . One Direction performed " Kiss You " on both the UK and US versions of The X[[ Factor]]
  - -3.09  @-@ out show at New York City 's Madison Square Garden . One Direction delivered a performance of " Kiss You " , in front of a video game @-@ themed set , on the final of the ninth series of The X[[ Factor]]
  - -2.65  Mail , their " energetic rendition " of " Kiss You " proved that the group have an elusive quality . On 12 December 2012 , the group also performed the number on the final of the season of The X[[ Factor]]
  - -1.82  where they loaded supplies from for the victims on 4 September . Together with two other battleships and a pair of light cruisers , Ise sank the destroyer Yayoi in 1926 during gunnery practice . Ise[[ ']]
  - -1.81 romachi period , e.g. , Taiheiki , a historical epic in the late 14th century , and some noh plays such as Hyakuman , Bashō , and Shunkan .    During the Kan[[ ']]
  - -1.80  light , often finely observed " ; while the poems from the late Kuizhou period have a " density and power of vision " .    Although he wrote in all poetic forms , Du Fu is best known for his lǜshi[[ ,]]
  - -1.80  to the MHSAA 's cancellation of state championships in all sports in 1943 .    = = High school = =    Rifenburg was born in Petoskey , Michigan , and raised in Kalamazoo[[ ,]]
  - -1.80  rather than the " reductive " categorizations used for more limited poets .    = = = Early years = = =    Most of what is known of Du Fu 's life comes from his poems . His paternal grandfather was Du Shenyan[[ ,]]
Negative pole, set B:
  - -2.81 ed covered the song " Ghosts of War " for their 2009 cover album For the Lions . They released a music video for it also . Korn has covered the title track at least twice live , once with Kid[[ Rock]]
  - -2.06  Slayer 's first record to do so in that country . Slayer 's official biography states that " some critics praised the album as demonstrating Slayer 's desire to grow musically and avoid repeating themselves . " Alex Henderson of All[[Music]]
  - -1.89  Egyptian ones . Egyptian cults sometimes incorporated Greek language , philosophy , iconography , and even temple architecture . Meanwhile , the cults of several Egyptian deities — particularly Isis , Osiris , Anubis , the form of Horus named Harpocrates[[ ,]]
  - -1.87  queen and creator goddess . Newly prominent gods tended to adopt characteristics from their predecessors . Isis absorbed the traits of many other goddesses during her rise , and when Amun became the ruler of the pantheon , he was conjoined with Ra[[ ,]]
  - -1.86  , both of which were personified as goddesses .    Nationally important deities gave rise to local manifestations , which sometimes absorbed the characteristics of older regional gods . Horus had many forms tied to particular places , including Horus of Nekhen[[ ,]]
Writes up (raises the score of): [' and', ',', ' ', ' in', '.', '-', ' (', ' is', '\xa0', ' for', ' I', ' of']
Writes down (lowers the score of): ['<filename>', ' strugg', ' affili', ' comprom', ' necessit', ' enthus', 'Intf', ' dissemin', 'firehose', ' policym', ' interpre', ' psychiat']
Zeroing it: mean loss change +0.0033 nats. Tokens hurt most:
  - +5.63  2012 , to a record crowd estimated at 15 @,@ 000 . " Kiss You " was included in the set list of the group 's 3 December 2012 sold @[[-]]
  - +4.41  and near missed multiple times ; all told she lost 50 crewmen killed and many others wounded . The bombs started numerous leaks and Ise began to settle by the bow , although she was returned to an even keel after three @[[-]]
  - +3.33  Girl " , Elvis Presley 's " Blue Hawaii " and Rammstein 's " Mein Land " , among others .    The music video garnered 10 @.@ 4 million views in a 4 @[[-]]
  - +3.33  10 @.@ 4 million views in a 4 @-@ hour period , failing to attain the Vevo record held by Justin Bieber 's " Beauty and a Beat " music video ( 10[[ @]]
  - +2.99  Li Po ) for the first time , and the two poets formed a friendship . David Young describes this as " the most significant formative element in Du Fu 's artistic development " because it gave him a living example of the reclusive poet @[[-]]
  - +2.97  submarine .    In early November the catapults were removed from both ships , and they loaded troops and munitions later that month . While en route they were diverted to the Spratly Islands upon reports of heavy air raids at Manila . After @[[-]]

## O378 (carries part of T2)

Positive pole, set A:
  - +2.02  and in the aristocratic society , and as a result poetry was often cited in Japanese literature in the Muromachi period , e.g. , Taiheiki , a historical epic in the late 14th century , and some no[[h]]
  - +1.27  's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . Boulter starred as " Darren " , in the 2005 theatre productions of the Philip[[ Rid]]
  - +1.27  Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip[[ Rid]]
  - +1.06  Diet ( parliament ) to secure funding for the entirety of the program at once . Because of economic constraints , only four battlecruisers and a single battleship of the Fusō class were ultimately approved by the Diet . Three more Fusō @[[-]]
  - +1.06  . This reduced the crew 's accommodations despite a significant increase in the crew 's numbers and naval historian Fukui Shizuo believed that these ships had the worst habitability of any Japanese capital ship . The final design was designated A @[[-]]
  - +1.04  the necessary accommodations . His life , like the whole country , was devastated by An Lushan Rebellion of 755 , and his last 15 years were a time of almost constant unrest .    Although initially he was little @[[-]]
  - +1.01 500 nautical miles ( 17 @,@ 600 km ; 10 @,@ 900 mi ) at a speed of 16 knots . A pair of Type 22 surface @[[-]]
  - +0.98 14 .    = = Design and description = =    The progress of Fusō 's construction , while the IJN waited for the funding to be released and foreign developments , caused the IJN to reassess the Fusō @[[-]]
Positive pole, set B:
  - +1.04  and private offerings are generally called " personal piety " : acts that reflect a close relationship between an individual and a god . Evidence of personal piety is scant before the New Kingdom . Votive offerings and personal names , many of which are theop[[hor]]
  - +0.99  pilots were trained . During World War II civil aerodromes were taken over for military use , existing military airfields were expanded , and new ones were built . This resulted in a significant inventory of facilities becoming available after the war . Pre @[[-]]
  - +0.98 ty ; and with several solar deities as Horemakhet @-@ Khepri @-@ Ra @@ Atum . On rare occasion , even deities of different sexes were joined in this way , producing combinations like Osiris @[[-]]
  - +0.95  per cent are engaged in GA operations , and annually the GA fleet accounts for between 1 @.@ 25 and 1 @.@ 35 million hours flown . The single most common class of aircraft is the fixed @[[-]]
  - +0.95  , developed from the late 1940s Cessna 170 , was introduced in 1956 , and became the world 's best selling single @-@ engine aeroplane . Single piston @[[-]]
Negative pole, set A:
  - -2.91  he is frequently called the greatest of the Chinese poets . His greatest ambition was to serve his country as a successful civil servant , but he proved unable to make the necessary accommodations . His life , like the whole country , was devastated by An L[[ush]]
  - -2.18  minor post , in normal times it would have been at least the start of an official career . Even before he had begun work , however , the position was swept away by events .    = = = War = = =    The An L[[ush]]
  - -1.87  remaining rivals for imperial dominance in the Pacific Ocean : Britain and the United States . Satō Tetsutarō , a Japanese Navy admiral and military theorist , speculated that conflict would inevitably arise between Japan and at least one of its two[[ main]]
  - -1.46  ) .    = = Ships = =    = = Service = =    Upon commissioning , the sister ships were assigned to the 1st Battleship Division of the 1st Fleet . Hyūga had an explosion in one of her[[ main]]
  - -1.20  who appreciated its carefree , jubilant nature .    The song was included in the dancing game Just Dance 2014 , and is also one of the select songs available on the demo version . Additionally , it is the final[[ main]]
  - -1.19  9th century . The first notable Japanese appreciator of Du Fu 's poetry was Kokan Shiren ( 1278 – 1346 ) , a Rinzai Zen patriarch and one of the[[ most]]
  - -1.03 30s modernization , except for No. 6 turret as its supporting structure could not be lowered . The recoil mechanism of the guns was also changed from a hydraulic to a pneumatic system , which allowed for a faster firing cycle of the[[ main]]
  - -0.84 5 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Sh[[ane]]
Negative pole, set B:
  - -1.74  could write " slow material as well . " Metal Forces reviewer gives " the band credit for at least making an effort to try something new and not being afraid to experiment at such a crucial stage of their career " , creating " one of the[[ more]]
  - -1.18  latest periods of Egyptian history . These later eras saw more religious innovations , including the practice of giving animal mummies as offerings to deities depicted in animal form , such as the cat mummies given to the feline goddess Bastet . Some of the[[ major]]
  - -1.13  1986 's Reign in Blood . Lombardo felt it made Slayer seem as though they " had matured a little bit " , while Friedman himself deemed it " a really cool back cover " and " one of the[[ most]]
  - -1.07  fleet accounts for between 1 @.@ 25 and 1 @.@ 35 million hours flown . The single most common class of aircraft is the fixed @-@ wing light aircraft associated with traditional GA , but the[[ main]]
  - -1.06  public transport operations , and generally does favour the development of smaller aerodromes catering to the GA market . The planning process has become a mechanism for addressing local aerodrome @-@ related environmental issues which , particularly regarding noise , are the[[ main]]
Writes up (raises the score of): ['RIG', 'ploys', 'theless', ' ecologically', ' comprom', ' _,', ' strugg', 'leness', 'ger', 'WARE', '’?', ' encomp']
Writes down (lowers the score of): ['istle', 'ordial', 'uckingham', 'sectional', 'asphem', ' evil', 'ifs', ' tail', 'assertIn', 'overning', 'on', ' sample']
Zeroing it: mean loss change +0.0007 nats. Tokens hurt most:
  - +0.38 012 , to a record crowd estimated at 15 @,@ 000 . " Kiss You " was included in the set list of the group 's 3 December 2012 sold @-@[[ out]]
  - +0.30 acoub . " Kiss You " is an upbeat power pop song with electronic effects ; the lyrics detail a protagonist 's infatuation with a significant other . Critics praised the song for its production , calling it a stand @-@[[ out]]
  - +0.30  @,@ 000 ft ) at an elevation of 85 degrees . The maximum effective rate of fire was only between 10 and 120 rounds per minute because of the frequent need to change the fifteen @[[-]]
  - +0.29  even keel after three @-@ days pumping . Hyūga was a primary focus of attack and she received 10 direct hits and up to 30 near misses . She was badly damaged with some 200 @[[-]]
  - +0.25 ter portrayed a character named " Sean " in Donkey Punch , who tags along with character " Josh " as the " quiet brother ... who hits it off with Tammi " . Boulter guest starred on a two @[[-]]
  - +0.25 @ 700 yd ) at 37 knots ( 69 km / h ; 43 mph ) .    In1931 – 33 the AA guns were replaced with eight 40 @[[-]]

(T0: parent pole +, 12 coverage children, children explain 98% of its top tokens)

## T0

Positive pole, set A:
  - +3.01  number 152 in the UK Singles Chart on 24 November 2012 . It peaked at number nine on the UK Singles Chart on 26 January 2013 , becoming One Direction '[[s]]
  - +2.94  resulting fire and save the ship . She received temporary repairs during which the turret was removed and replaced by a circular armour plate on which three triple 25 mm gun mounts were positioned . On 11 May a valve in Ise '[[s]]
  - +2.78  Hot 100 , peaking at number 30 . The single bowed at number 13 on the Australian Singles Chart on 27 January 2013 , marking its peak position and the group '[[s]]
  - +2.76  of 3 : 04 ( 3 minutes , four seconds ) . The track features electronic effects , colossal hooks , a " na na na " breakdown , and a Motown @-@ tinged melody . One Direction '[[s]]
  - +2.71  was certified gold by the Recording Industry Association of America ( RIAA ) on 25 April 2013 , denoting shipments of 500 @,@ 000 copies .    The song became One Direction '[[s]]
  - +2.68 ian Lundin , Albin Nedler , Savan Kotecha , Shellback , and its producers , Carl Falk and Rami Yacoub . Falk , Kotecha , and Yacoub had collaboratively composed One Direction '[[s]]
  - +2.67  Norway and Denmark associations , signifying collective shipments of 20 @,@ 000 units .    = = Music video = =    The accompanying music video , directed by Vaughan Arnell , who had previously directed One Direction '[[s]]
  - +2.67  's previous hit singles , " What Makes You Beautiful " , " One Thing " , and " Live While We 're Young " . In April 2012 , The Independent reported that Simon Cowell , the group '[[s]]
Positive pole, set B:
  - +2.89  Shu . For these reasons , the Egyptologist Dominic Montserrat suggested that Akhenaten may have been monolatrous , worshipping a single deity while acknowledging the existence of others . In any case , Atenism '[[s]]
  - +2.86  British Columbia during 2006 's Unholy Alliance tour .    " Behind the Crooked Cross " is rarely played live as Hanneman hates the track , though King has always wanted to play it " because it '[[s]]
  - +2.68  type of operation is the transport by helicopter of spectators to the British Formula One grand prix at Silverstone . This involves so many flights that , according to Cranfield Aviation Services , on race day the heliport is temporarily the world '[[s]]
  - +2.66  Yet Atenism may not have been full monotheism , which totally excludes belief in other deities . There is evidence suggesting that the general populace was still allowed to worship other gods in private . The picture is further complicated by Atenism '[[s]]
  - +2.65  his regular vocal style , and " added too much singing . " Drummer Dave Lombardo has since observed : " There was fire on all the records , but it started dimming when South of Heaven came into the picture . And that '[[s]]
Negative pole, set A:
  - -3.68  .    The first battleships built for the renewed Eight @-@ Eight Fleet Program were the two dreadnoughts of the Kawachi class , ordered in 1907 and laid down in 1908 . In[[ ]]
  - -3.58  School in football , basketball , and track and . In 1943 , Michigan canceled boys high school tournaments in all sports due to World War II , and they did not return until the fall of 1944 . In[[ ]]
  - -3.58  general at Chengdu . Despite his financial problems , this was one of the happiest and most peaceful periods of his life . Many of Du 's poems from this period are peaceful depictions of his life at " thatched hut " . In[[ ]]
  - -3.58  role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In[[ ]]
  - -3.49  again in 755 . He married around 752 , and by 77 the couple had had five children — three sons and two daughters — but one of the sons died in infancy in 755 . From[[ ]]
  - -3.46  Fu was by some years the younger , while Li Bai was already a poetic star . We have twelve poems to or about Li Bai from the younger poet , but only one in the other direction . They met again only once , in[[ ]]
  - -3.45  5 / + 20 degrees . The guns had a rate of fire of 1 @.@ 5 – 2 rounds per minute and could be loaded at any angle between -3 and + 20 degrees . In[[ ]]
  - -3.36  " , and had a role as a different character " Toby Steele " on The Bill .    He had a recurring role in 2003 on two episodes of The Bill , as character " Connor Price " . In[[ ]]
Negative pole, set B:
  - -3.79  She was assigned to the III Battleship Division .    For most of World War I , Erzherzog Ferdinand Max remained in her home port of Pola , in present @-@ day Croatia , except for four engagements . In[[ ]]
  - -3.76  Club at Muswell Manor on the Isle of Sheppey , and in May 1909 it was the venue of the first flight conducted in the country by a British pilot , John Moore @-@ Brabazon . In[[ ]]
  - -3.41  , Modest Mouse and Califone , Pro @-@ Pain , and Universe Eye . Polish death metal band Decapitated covered the song " Mandatory Suicide " on their first full @-@ length album Winds of Creation . In[[ ]]
  - -3.38  was one of only two Def Jam titles to be distributed by Geffen Records through Warner Bros. Records because of original distributor Columbia Records ' refusal to release work by the band . The release peaked at number 57 and in[[ ]]
  - -3.29  to become a car dealer after his hockey career , as his father , uncles and brother all worked in the industry . After apprenticing under his brother for five years , he bought a Dodge dealership in Newnan , Georgia in[[ ]]
Writes up (raises the score of): ['Intf', 'firehose', ' affili', ' strugg', 'MetaInfo', '<filename>', 'vantages', 'ChannelType', ' psychiat', ' accred', 'CreateModel', ' Ishma']
Writes down (lowers the score of): [' and', ',', '.', ' (', '-', ' ', ' in', ':', ' is', ' I', ';', ' a']
Zeroing it: mean loss change +0.0022 nats. Tokens hurt most:
  - +2.44  excellence = = =    Du Fu 's work is notable above all for its range . Chinese critics traditionally used the term 集大成 ( jídàchéng- " complete symphony " ) , a reference to Men[[cius]]
  - +1.37  colloquial to the allusive self @-@ consciously literary . This variety is manifested even within individual works : Owen identifies the , " rapid stylistic and thematic shifts " in poems which enable the poet to represent different facets of a situation , while Ch[[ou]]
  - +1.26 80 km ) ahead of the American carriers while waiting for the Japanese ships to be located . This was just the first of a total of five airstrikes that the Americans launched that day . The sisters were not heavily engaged by the[[ early]]
  - +1.24  a 2006 interview , fellow actor Ben Whishaw identified Boulter as one of his favorite co @-@ stars : " I loved working with a guy called Robert Boulter , who was in the triple bill[[ of]]
  - +1.23  intended aircraft . Coupled with a shortage of trained pilots , neither ship ever used its aircraft during combat .    = = = Other changes = = =    After the loss of the fast battleship Hiei at the Naval Battle of Gu[[adal]]
  - +1.03  @-@ search radars were installed aboard the sisters in mid @-@ 1942 .    = = = Aircraft = = =    Ise was briefly fitted with an aircraft flying @-@ off platform for a Mits[[ub]]

## O279 (carries part of T0)

Positive pole, set A:
  - +2.63  Pretending you know what you 're talking about when you don 't is very bizarre but there are advisers on set who are fantastic at taking you through procedures and giving you the confidence to stand there and look like you know what you '[[re]]
  - +2.46  Falk and Rami Yacoub . Falk , Kotecha , and Yacoub had collaboratively composed One Direction 's previous hit singles , " What Makes You Beautiful " , " One Thing " , and " Live While We '[[re]]
  - +2.38  kind of romp " , as noted by Arnell in a MTV News interview .    Shot by November 2012 , the music video was characterised , in several MTV News interviews , as " bigger than anything we '[[ve]]
  - +2.37  20 @,@ 000 units .    = = Music video = =    The accompanying music video , directed by Vaughan Arnell , who had previously directed One Direction 's music videos for " Live While We '[[re]]
  - +2.37  wrote that for Du Fu , " everything in this world is poetry " , Du wrote extensively on subjects such as domestic life , calligraphy , paintings , animals , and other poems .    = = = Technical excellence = = =    Du Fu '[[s]]
  - +2.37  as " Kieron Fletcher " . He portrayed an emergency physician applying for a medical fellowship . He commented on the inherent difficulties in portraying a physician on television : " Playing a doctor is a strange experience . Pretending you know what you '[[re]]
  - +2.25  Kristian Lundin — writing    Albin Nedler — writing , background vocals    Shellback — writing    Rami Yacoub — writing , production , programming , instruments , bass    Credits adapted from Take Me Home '[[s]]
  - +2.19  variety of the poet 's work required consideration of his whole life , rather than the " reductive " categorizations used for more limited poets .    = = = Early years = = =    Most of what is known of Du Fu '[[s]]
Positive pole, set B:
  - +3.03  and not being afraid to experiment at such a crucial stage of their career " , creating " one of the more original sounding thrash / speed metal albums he heard in a long while " . He remarks , however , that " if you ’[[ re]]
  - +2.18 ala @-@ la @-@ la @-@ la . ' I can 't see myself playing it , but after that , where it gets heavier , I like that section . If we ever did a medley , I '[[d]]
  - +2.16  only cover version to appear on a Slayer studio album . The song was chosen due to its war @-@ themed lyrics . Hanneman described the track as " more just like one of those odd songs that a lot of people didn '[[t]]
  - +2.15  to the Apocalypse , Adrien Begrand of PopMatters described the album as " their most underrated , and on this set , its five selections show how highly the band thinks of the record . " KNAC.com '[[s]]
  - +2.13  that he hates the track : " That 's one of the black marks in our history , in my book . I just fucking think it 's horrible . [ Laughs ] I hate the opening riff . It '[[s]]
Negative pole, set A:
  - -2.66 0 t ) at deep load , roughly 650 long tons ( 660 t ) more than the preceding class . Their crew consisted of 1 @,@ 360 officers and enlisted men . They had a[[ met]]
  - -2.30  Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip[[ Rid]]
  - -2.25  's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . Boulter starred as " Darren " , in the 2005 theatre productions of the Philip[[ Rid]]
  - -1.99  radars were also fitted during the conversion .    The removal of the secondary armament , the rear turrets and their supporting structures was generally compensated by the addition of the flight deck , hangar , AA guns and more fuel , and the[[ met]]
  - -1.77  you just wanna take me home / Baby say yeah , yeah , yeah , yeah , yeah . "    " Kiss You " was well received by contemporary music critics , who centred on its quality of production . Both Rolling Stone 's Jon[[ Dol]]
  - -1.67  40 , some 200 miles ( 320 km ) east of Cape Engaño , the northeastern tip of Luzon . The American carriers were spread out and it was very late in the day to launch an[[ a]]
  - -1.56  , the group 's manager , had challenged prominent songwriters to compete for space on One Direction 's second album . Falk said , " It 's important to get their personalities on the music . " In addition , the article reported that[[ Sy]]
  - -1.55  was released as the record 's second single in Germany and the third overall single on 7 January 2013 . The song was composed by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler ,[[ Sav]]
Negative pole, set B:
  - -1.70  instance , Khnum was the god of Elephantine Island in the midst of the Nile , the river that was essential to Egyptian civilization . He was credited with producing the annual Nile flood that fertilized the nation 's farmland . Perhaps as an[[ out]]
  - -1.65  aircraft , such as microlights , amateur built aeroplanes , and smaller helicopters . There are 28 @,@ 000 Private Pilot Licence holders , and 10 @,@ 000 certified[[ gl]]
  - -1.64  Erzherzog Ferdinand Max and her sisters were stationed in Cattaro in their place . On the morning of 11 June , Admiral Miklos Horthy planned a major assault on the Otranto Barrage ; the three[[ Er]]
  - -1.64  Max took a minor part in the bombardment of Ancona on 24 May 1915 . There she and her sisters expended 24 rounds of 240 mm armor @-@ piercing shells at signal and[[ sem]]
  - -1.62  came with the group 's 2007 re @-@ release of ninth studio album Christ Illusion , shot in Vancouver , British Columbia during 2006 's Unholy Alliance tour .    " Behind the[[ Cro]]
Writes up (raises the score of): [' strugg', ' affili', ' enthus', ' comprom', ' dissemin', ' psychiat', ' authent', ' acknow', ' dismant', ' interspers', ' accred', ' necessit']
Writes down (lowers the score of): ['-', ',', '.', ' (', ' ', ' and', '/', '\n', '\xa0', ' is', ':', ' in']
Zeroing it: mean loss change +0.0003 nats. Tokens hurt most:
  - +0.73 011 film Mercenaries directed by Paris Leonti .    = = Filmography = =    = = = Film = = =    = = = Television = = =    = = = Theatre = = =    = Du[[ Fu]]
  - +0.58 enaries directed by Paris Leonti .    = = Filmography = =    = = = Film = = =    = = = Television = = =    = = = Theatre = = =    = Du Fu =    Du Fu[[ (]]
  - +0.56 @ 000 digital downloads in the US by 18 January 2013 . The single ultimately peaked at number 46 on the Hot 100 and was certified gold by the Recording Industry Association of America[[ (]]
  - +0.47  , and as a result poetry was often cited in Japanese literature in the Muromachi period , e.g. , Taiheiki , a historical epic in the late 14th century , and some noh plays such as Hy[[ak]]
  - +0.39  misses that put her on the bottom in shallow water with a 15 degree list . The sisters were struck off the Navy List in November and their wrecks were scrapped after the war .    = Dick Rifenburg =    Richard Gale[[ "]]
  - +0.38 in , Albin Nedler , Savan Kotecha , Shellback and its producers , Carl Falk and Rami Yacoub . " Kiss You " is an upbeat power pop song with electronic effects ; the lyrics detail a[[ protagonist]]

(T1: parent pole +, 11 coverage children, children explain 100% of its top tokens)

## T1

Positive pole, set A:
  - +6.90  aunt . He had an elder brother , who died young . He also had three half brothers and one half sister , to whom he frequently refers in his poems , although he never mentions his stepmother .    The son of a scholar @-[[@]]
  - +6.63  in the manually loaded 152 @-@ millimetre ( 6 in ) secondary guns used in the Fusō class and earlier . To resolve this issue , the IJN designed a smaller 140 @-[[@]]
  - +6.50  metres per second ( 2 @,@ 640 ft / s ) . A special Type 3 Sanshikidan incendiary shrapnel shell was developed in the 193s for anti @-[[@]]
  - +6.44  necessary accommodations . His life , like the whole country , was devastated by An Lushan Rebellion of 755 , and his last 15 years were a time of almost constant unrest .    Although initially he was little @-[[@]]
  - +6.41  of earlier poets took place , in which Wang Wei , Li Bai and Du Fu came to be regarded as representing respectively the Buddhist , Daoist and Confucian strands of Chinese culture . At the same time , the development of Neo @-[[@]]
  - +6.40 9 @,@ 060 mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    = = = Armament = = =    The twelve 45 @-[[@]]
  - +6.35  special Type 3 Sanshikidan incendiary shrapnel shell was developed in the 193s for anti @-@ aircraft use .    The ships ' secondary armament consisted of twenty 50 @-[[@]]
  - +6.33 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn . In May 2008 , Boulter made a guest appearance on a two @-[[@]]
Positive pole, set B:
  - +6.66  ; and with several solar deities as Horemakhet @-@ Khepri @-@ Ra @@ Atum . On rare occasion , even deities of different sexes were joined in this way , producing combinations like Osiris @-[[@]]
  - +6.52  . He played his first game in nearly 18 months on September 18 , 198 , in an exhibition game between the Lightning and Sabres at Innsbruck , Austria . Cullen scored the game @-[[@]]
  - +6.33  Industry Association of America .    In order to offset the pace of the group 's previous album , Slayer deliberately slowed down the album 's tempo . In contrast to their previous , the band utilized undistorted guitars and toned @-[[@]]
  - +6.32 ome ( province ) of Egypt . Some of these groups contain a specific , symbolically important number of deities . Paired gods can stand for opposite but interrelated concepts that are part of a greater unity . Ra , who is dynamic and light @-[[@]]
  - +6.29  on standards of airworthiness and pilot licensing , and the objective is to promote high standards of safety . At the lighter end of the GA spectrum some regulatory authority is devolved to representative bodies , and gliding is in transition from a self @-[[@]]
Negative pole, set A:
  - -1.56  =    Richard Gale " Dick " Rifenburg ( August 21 , 1926 – December 5 , 1994 ) was an American football player and a pioneering television broadcaster for the forerunner to W[[IV]]
  - -1.54  in 1922 during the prince 's visit to Japan . In Korea Bay when the 1923 Great Kantō earthquake struck , they sailed to Kyushu where they loaded supplies from for the victims on 4[[ September]]
  - -1.46 orage at Hashirajima to join the unsuccessful search for the American carrier force that had attacked Marcus Island a week earlier . Similarly they pursued but did not catch the American carriers that had launched the Doolittle Raid on 18[[ April]]
  - -1.44  until the actual conversions began . Ise was converted at Kure Naval Arsenal from 23 February to 5 September 1943 and Hyūga at Sasebo Naval Arsenal from 2 May 30[[ November]]
  - -1.44  Battleship Division set sail with the Aleutian Support Group on 28 May , at the same time that most of the Imperial Fleet began an attack on Midway Island ( Operation MI ) .    They returned home on 14[[ June]]
  - -1.44 año = = = =    Shortages of aircraft and serviceability problems greatly retarded pilot training and the ships only had a total of 17 D4Ys and 18 E16As on hand on 1[[ October]]
  - -1.44  the sisters and their escorts to bring a load of petrol , rubber , tin and other strategic minerals back to Japan after the American carriers departed the South China Sea ( Operation Kita ) . They loaded their cargoes beginning on 6[[ February]]
  - -1.42  search for the American carrier force that had attacked Marcus Island a week earlier . Similarly they pursued but did not catch the American carriers that had launched the Doolittle Raid on 18 April .    During gunnery training on 5[[ May]]
Negative pole, set B:
  - -1.47  in the naval base , the armored cruisers Sankt Georg and Kaiser Karl VI were decommissioned and Erzherzog Ferdinand Max and her sisters were stationed in Cattaro in their place . On the morning of 11[[ June]]
  - -1.47  " ) was a pre @-@ dreadnought battleship built by the Austro @-@ Hungarian Navy in 1902 . The second ship of the Erzherzog Karl class , she was launched on 3[[ October]]
  - -1.46  , the river that was essential to Egyptian civilization . He was credited with producing the annual Nile flood that fertilized the nation 's farmland . Perhaps as an outgrowth of this life @-@ giving function , he was said to create all living[[ things]]
  - -1.44  in south eastern Italy , the Austro @-@ Hungarian ships were recalled . In company with other units of the Austro Hungarian navy , Erzherzog Ferdinand Max took a minor part in the bombardment of Ancona on 24[[ May]]
  - -1.44  fleet by luring Allied ships to the cruisers and lighter ships , which were protected from the heavier guns of the battleships , including the guns of the Erzherzog Karl class . However , on the morning of 10[[ June]]
Writes up (raises the score of): ['MetaInfo', 'firehose', 'Intf', 'ChannelType', ',,,,', '?](', '<filename>', ',,,,,,,,', 'insic', 'CreateModel', ' strugg', ' dilig']
Writes down (lowers the score of): [' in', ' is', ',', '.', ' was', ' to', ' I', ' for', ' (', ':', ' ', ' set']
Zeroing it: mean loss change +0.0035 nats. Tokens hurt most:
  - +2.02 2012 , to a record crowd estimated at 15 @,@ 000 . " Kiss You " was included in the set list of the group 's 3 December 2012 sold @-[[@]]
  - +1.57 fer Girl " , Elvis Presley 's " Jailhouse Rock " and Rammstein 's " Mein Land " . The music video received 10 @.@ 4 million views in a 24 @[[-]]
  - +1.56  also added and the turret roofs were increased to a total of 152 millimetres ( 6 in ) of armour .    = = = Fire control and sensors = = =    While the details of the ship 's fire @[[-]]
  - +1.48 ing it sonically . A reviewer for MTV News described the track 's lyricism as " butterflies @-@ inducing " , and Sam Lansky of Idolator wrote that " Kiss You " is noticeably a stand @-[[@]]
  - +1.48  , but who has the natural speed , smooth actions , height and big hands that is the mark of an outstanding pass receiver . Rifenburg has laid claim to that rating . A loose @-@ limbed 180 @-@[[ pound]]
  - +1.38 008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn . In May 2008 , Boulter made a guest appearance on a two @-@[[ part]]

## O436 (carries part of T1)

Positive pole, set A:
  - +0.93  largely ceremonial . Du Fu 's conscientiousness compelled him to try to make use of it : he caused trouble for himself by protesting the removal of his friend and patron Fang Guan on a petty charge . He was arrested but was p[[ard]]
  - +0.92  of Empress Wu . Du Fu was born in 712 ; the exact birthplace is unknown , except that it was near Luoyang , Henan province ( Gong county is a favourite candidate ) . In later life , he considered[[ himself]]
  - +0.91  of his half brothers . He spent the next four years living in the Luoyang area , fulfilling his duties in domestic affairs .    In the autumn of 744 , he met Li Bai ( Li Po ) for the first[[ time]]
  - +0.79  from the Muromachi period and on scholars and poets in the Edo period , including Matsuo Bashō , the very greatest of all haiku poets . Even in modern Japanese , the term Saint of Poetry ( 詩聖[[ ,]]
  - +0.78  as Hyakuman , Bashō , and Shunkan .    During the Kan 'ei era of the Edo period ( 1624 – 1643 ) , Shào Chuán ( 邵�[[�]]
  - +0.77  two other music videos . The clip depicts the band shooting various scenes via a green screen , which include sequences reminiscent of iconic music videos of songs such as the Beach Boys ' " Surfer Girl " , Elvis Presley 's " J[[ail]]
  - +0.75  and what he hoped for or feared from the progress of various campaigns — these became the enduring themes of his poetry " . Even when he learned of the death of his youngest child , he turned to the suffering of others in his poetry instead of[[ dwelling]]
  - +0.71  it a stand @-@ out track on Take Me Home .    The track became the group 's sixth top @-@ ten hit in Ireland and the United Kingdom , while attaining top @-@ forty positions in both Belgian territories ([[ Flanders]]
Positive pole, set B:
  - +1.26  renamed its Comeback Player of the Year award the John Cullen Award .    Former Lightning head coach Terry Crisp has stated publicly that Cullen was a player that stood out as something special saying “ John Cullen ... beat cancer and came[[ back]]
  - +1.04  and in one game against the Chicago Wolves tied an IHL record when he scored seven points in a 7 – 3 victory .    However , a bout of bronchitis led Cullen to fear that his cancer had returned . Tests came[[ back]]
  - +0.93  Isis poisons the superior god Ra and refuses to cure him unless he reveals his secret name to her . Upon learning the name , she tells it to her son , Horus , and by learning it they gain greater knowledge and power .    In[[ addition]]
  - +0.88  cover version to appear on a Slayer studio album . The song was chosen due to its war @-@ themed lyrics . Hanneman described the track as " more just like one of those odd songs that a lot of people didn 't[[ know]]
  - +0.85 er 's Karl Willetts both rate South of Heaven as amongst the top five albums of all time , while Max Kolesne of Brazilian death metal group Krisiun remembers hearing the song " Silent Scream " for the first[[ time]]
Negative pole, set A:
  - -1.78 okosuka E1Y reconnaissance floatplane in 1928 – 29 . A catapult and a collapsible 4 @- tonne ( 3 @.@ 9 @-@ long @-[[@]]
  - -1.70 5 for a Yokosuka E1Y reconnaissance floatplane in 1928 – 29 . A catapult and a collapsible 4 @- tonne ( 3 @.@ 9 @-[[@]]
  - -1.56  @-@ aircraft defence was provided by four 40 @-@ calibre 3rd Year Type 8 @-@ centimetre AA guns in single mounts . The 7 @.@ 62 @-[[@]]
  - -1.56  the midships magazine and exposed more of the ship to the blast effects of the guns when they fired . Another issue was that Japanese sailors had problems maintaining a high rate of fire with the 45 @.@ 36 @-[[@]]
  - -1.51 931 – 33 the AA guns were replaced with eight 40 @-@ caliber 12 @.@ 7 cm ( 5 @.@ 0 in ) Type 89 dual @-[[@]]
  - -1.50     = = = Propulsion = = =    The Ise @-@ class ships had two sets of direct @-@ drive steam turbines , each of which drove two propeller shafts with 3 @.@ 429 @-[[@]]
  - -1.50  ) to a maximum height of 7 @,@ 500 metres ( 24 @,@ 600 ft ) . The ships were also fitted with six submerged 53 @.@ 3 @-[[@]]
  - -1.50  metres ( 17 @,@ 800 yd ) . Each gun had a rate of fire of up to 10 rounds per minute . Anti @-@ aircraft defence was provided by four 40 @-[[@]]
Negative pole, set B:
  - -1.45  centres , he scored 94 points in the Penguins ' first 65 games and played in his first NHL All @-@ Star Game . However , when Lemieux returned after missing an additional 50 @-[[@]]
  - -1.36 7 in ) torpedo tubes , although rarely used .    = = Service history = =    At the outbreak of World War I , Erzherzog Ferdinand Max was in the III division of the Austrian @-@ Hungarian battle @-[[@]]
  - -1.32  the British 24 cm / 40 ( 9 @.@ 4 " ) Krupp C / 94 , which was used on the Habsburgs . Her secondary armament consisted of twelve 19 @-[[@]]
  - -1.29  Horemakhet @-@ Khepri @-@ Ra @@ Atum . On rare occasion , even deities of different sexes were joined in this way , producing combinations like Osiris @-@ Neith and Mut @-[[@]]
  - -1.29 @ 6 centimetres ( 2 @.@ 6 in ) / 45 caliber gun , also manufactured by Škoda . Anti @-@ aircraft and airship protection was covered by the four 37 @-[[@]]
Writes up (raises the score of): [' Fitzgerald', 'ortunate', ' basket', 'metics', 'ele', ' Fairy', ' freed', ' Movement', 'ULD', ' Walker', ' Extension', ' trough']
Writes down (lowers the score of): ['щ', 'ronomy', 'nec', 'Subthreshold', 'angelo', 'owl', "('.')", 'aku', 'inker', 'AddField', 'reshold', 'nings']
Zeroing it: mean loss change -0.0005 nats. Tokens hurt most:
  - +0.27  as they complicated the protection of the midships magazine and exposed more of the ship to the blast effects of the guns when they fired . Another issue was that Japanese sailors had problems maintaining a high rate of fire with the 45 @.[[@]]
  - +0.24 ualty in 2010 , as " Kieron Fletcher " . Boulter starred in the 2011 film Mercenaries directed by Paris Leonti .    = = Career = =    =[[ =]]
  - +0.22  10 @.@ 4 million views in a 4 @-@ hour period , failing to attain the Vevo record held by Justin Bieber 's " Beauty and a Beat " music video ( 10[[ @]]
  - +0.22  .    = = Charts = =    = = Certifications = =    = = Release history = =    = Ise @-@ class battleship =    The Ise @-@ class battleships ( 伊�[[�]]
  - +0.21 0 mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    = = = Armament = = =    The twelve 45 @-@ calibre 35[[ @]]
  - +0.21  tons ) to 40 @,@ 444 tonnes ( 39 @,@ 805 long tons ) . This also reduced the draught to 9 @.@ 03 metres ( 2[[9]]

(T3: parent pole -, 12 coverage children, children explain 61% of its top tokens)

## T3

Positive pole, set A:
  - +5.04  as Hyakuman , Bashō , and Shunkan .    During the Kan 'ei era of the Edo period ( 1624 – 1643 ) , Shào Chuán ( 邵�[[�]]
  - +3.56  largely ceremonial . Du Fu 's conscientiousness compelled him to try to make use of it : he caused trouble for himself by protesting the removal of his friend and patron Fang Guan on a petty charge . He was arrested but was p[[ard]]
  - +3.54  inducing " , and Sam Lansky of Idolator wrote that " Kiss You " is noticeably a stand @-@ out track on its parent album . Melinda Newman , writing for Hitix , regarded the song as " a b[[oun]]
  - +3.48 7 @,@ 900 long tons ( 18 @,@ 200 t ) and armed with ten 12 @-@ inch ( 30 @.@ 5 cm ) guns , D[[read]]
  - +3.27  = =    = = Certifications = =    = = Release history = =    = Ise @-@ class battleship =    The Ise @-@ class battleships ( 伊勢型戦�[[�]]
  - +3.25  tons ( 18 @,@ 289 t ) each . This was the genesis of the Eight @-@ Eight Fleet Program , the development of a cohesive battle line of sixteen capital ships .    The launch of HMS D[[read]]
  - +3.21  he probably failed because his prose style at the time was too dense and obscure , while Chou suggests his failure to cultivate connections in the capital may have been to blame . After this failure , he went back to traveling , this time around Sh[[and]]
  - +3.09  Bai ( Li Po ) for the first time , and the two poets formed a friendship . David Young describes this as " the most significant formative element in Du Fu 's artistic development " because it gave him a living example of the re[[clusive]]
Positive pole, set B:
  - +4.02  change — whether direct interaction with the gods was a new development or an outgrowth of older traditions . Egyptians now expressed their devotion through a new variety of activities in and around temples . They recorded their prayers and their thanks for divine help on st[[el]]
  - +3.43  Ancona . She also took part in suppressing a major mutiny among the crew members of several armored cruisers stationed in Caro between 1 – 3 February 1918 . She also attempted to break through the Ot[[rant]]
  - +3.23  Kaiser Karl VI were decommissioned and Erzherzog Ferdinand Max and her sisters were stationed in Cattaro in their place . On the morning of 11 June , Admiral Miklos Horthy planned a major assault on the Ot[[rant]]
  - +3.12  on their first full @-@ length album Winds of Creation . In 2003 , " Silent Scream " was covered by Children of Bodom for their album Hate Crew Deathroll in UK version . Hardcore P[[unk]]
  - +3.11  beliefs .    = = = Unity of the divine traditional religion = = =    Scholars have long debated whether traditional Egyptian religion ever asserted that the multiple gods were , on a deeper level , unified . Reasons for this debate include the practice of syn[[cret]]
Negative pole, set A:
  - -2.96 913 . While Yamashiro was laid down later that year , the IJN lacked the funding to proceed with the construction of Ise and Hyūga until the Diet authorized additional funding for the ships in July 19[[1]]
  - -2.92  battleship of the Fusō class were ultimately approved by the Diet . Three more Fusō @-@ class ships ( Yamashiro , Ise , and Hyūga ) were approved and all three were ordered in April 19[[1]]
  - -2.89  Originally intended to be repeats of the preceding Fusō class , they were redesigned before construction began . Both ships carried supplies for the survivors of the Great Kantō earthquake in 1923 . They were modernized in 19[[3]]
  - -2.85 ayoi in 1926 during gunnery practice . Ise 's AA armament was upgraded in 1931 and Hyūga 's two years later . The latter ship was modernized in 19[[3]]
  - -2.82  first battleships built for the renewed Eight @-@ Eight Fleet Program were the two dreadnoughts of the Kawachi class , ordered in 1907 and laid down in 1908 . In 19[[1]]
  - -2.80  of Cape Engaño in late 1944 , where they decoyed the American carrier fleet supporting the invasion of Leyte away from the landing beaches . Afterwards both ships were transferred to Southeast Asia ; in early 19[[4]]
  - -2.79  Ise 's AA armament was upgraded in 1931 and Hyūga 's two years later . The latter ship was modernized in 1934 – 36 and Ise in 19[[3]]
  - -2.78  . He was a consensus selection at end on the 1948 College Football All @-@ America Team . Rifenburg played professionally in the National Football League ( NFL ) with the Detroit Lions for one season in 19[[5]]
Negative pole, set B:
  - -2.92  to the III Battleship Division .    For most of World War I , Erzherzog Ferdinand Max remained in her home port of Pola , in present @-@ day Croatia , except for four engagements . In 19[[1]]
  - -2.92  German ships SMS Goeben and SMS Breslau from the British @-@ held Mediterranean ; she advanced as far as Brindisi before being recalled to her home port . Her sole combat engagement occurred in late May 19[[1]]
  - -2.91  at Italian gun @-@ batteries and other port installations .    A major mutiny among crews of the armored cruisers stationed in Cattaro , including Sankt Georg and Kaiser Karl VI , began on 1 February 19[[1]]
  - -2.91 well Manor on the Isle of Sheppey , and in May 1909 it was the venue of the first flight conducted in the country by a British pilot , John Moore @-@ Brabazon . In 19[[1]]
  - -2.88  , the Austro @-@ Hungarian ships were recalled . In company with other units of the Austro Hungarian navy , Erzherzog Ferdinand Max took a minor part in the bombardment of Ancona on 24 May 19[[1]]
Writes up (raises the score of): [' (', '\xa0', ' ', ' and', ',', ' or', ' but', '.', ' A', '-', ' I', ' is']
Writes down (lowers the score of): [' strugg', '<filename>', ' acknow', ' enthus', ' dissemin', ' necessit', ' authent', ' psychiat', ' acknowled', ' interspers', ' affili', ' comprom']
Zeroing it: mean loss change +0.0013 nats. Tokens hurt most:
  - +1.62 1 @.@ 75 metres ( 104 ft 2 in ) and their draft to 9 @.@ 45 metres ( 31 ft 0 in ) . Their displacement increased over 5 @[[,]]
  - +1.60 926 during gunnery practice . Ise 's AA armament was upgraded in 1931 and Hyūga 's two years later . The latter ship was modernized in 1934 – 3[[6]]
  - +1.38  150 kilowatts ( 200 hp ) capacity and two 250 @-@ kilowatt ( 340 hp ) turbo generators at 225 volts .    During their 1930[[s]]
  - +1.24  War , both ships frequently exercised off the coasts of the Soviet Union , Korea and China in addition to training in Japanese waters . Ise hosted Edward , Prince of Wales , and his aide @-@ de @-@ camp Lieutenant Louis Mount[[bat]]
  - +1.07  2 , 3 , and 5 were built with imported 6 @-@ metre ( 19 ft 8 in ) Bausch & Lomb rangefinders . These were felt to be inferior to the British Barr & St[[roud]]
  - +0.96 3 long tons ( 5 @,@ 195 t ) of fuel oil that gave them a range of 7 @,@ 870 nautical miles ( 14 @,@ 580 km ;[[ ]]

## O148 (carries part of T3)

Positive pole, set A:
  - +2.18  13 – 16 @.@ 9 kg / cm2 ( 1 @,@ 275 – 1 @,@ 657 kPa ; 185 – 240[[ psi]]
  - +2.03  was not attacked when the Americans returned four days later . This time it was Ise 's turn and she was struck 11 or more times with many near misses that put her on the bottom in shallow water with a 15[[ degree]]
  - +1.89  Gō water @-@ tube boilers at working pressures of 13 – 16 @.@ 9 kg / cm2 ( 1 @,@ 275 – 1 @,@ 657[[ k]]
  - +1.71  3 @,@ 300 yd ) , and an effective ceiling of 5 @,@ 500 metres ( 18 @,@ 000 ft ) at an elevation of 85[[ degrees]]
  - +1.60  of fire was around eight rounds per minute . Two twin @-@ gun mounts for license @-@ built Vickers two @-@ pounder light AA guns were also added . These guns had a maximum elevation of + 80[[ degrees]]
  - +1.55  class ships were selected for the conversion because Hyūga had suffered an explosion in Turret No. 5 in early May that virtually destroyed the turret and their Turret No. 6 could not elevate to the full + 43[[ degrees]]
  - +1.55  the frequent need to change the fifteen @-@ round magazines . In addition the forward pair of 14 cm guns in the forecastle were removed at this time and the maximum elevation of the remaining guns was increased to + 30[[ degrees]]
  - +1.53 8 @-@ centimetre AA guns in single mounts . The 7 @.@ 62 @-@ centimetre ( 3 in ) high @-@ angle guns had a maximum elevation of + 75[[ degrees]]
Positive pole, set B:
  - +1.12 20 @,@ 00 metres ( 22 @,@ 000 yd ) at maximum elevation with a muzzle velocity of 800 metres per second ( 2 @,@ 600[[ ft]]
  - +1.02  . She was propelled by two two @-@ shaft , four cylinder vertical triple expansion steam engines . On trials , they developed 18 @,@ 000 ihp ( 13 @,@ 423[[ kW]]
  - +0.88  6 in ) / 45 caliber gun , also manufactured by Škoda . Anti @-@ aircraft and airship protection was covered by the four 37 @-@ millimeter ( 1 @.@ 5[[ in]]
  - +0.86 koda , mounted in eight single casemates on either wing of the ship and two twin turrets on the centerline. shell 20 @,@ 00 metres ( 22 @,@ 000[[ y]]
  - +0.85 2 @.@ 1 tons and could fire three rounds per minute.The ships had a tertiary armament for protection against torpedo boats in the form of the 6 @.@ 6 centimetres ( 2 @.@ 6[[ in]]
Negative pole, set A:
  - -1.18  work into English . As Burton Watson remarks in The Selected Poems of Du Fu , " There are many different ways to approach the problems involved in translating Du Fu , which is why we need as many different translations as possible " ( p. x[[xi]]
  - -0.86  productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Z[[aza]]
  - -0.81  The Daily Telegraph : " The acting is shatteringly intense , with wired performances from Ben Whishaw ( now unrecognisable from his performance as Trevor Nunn 's Hamlet ) , Robert Boulter , Shane Z[[aza]]
  - -0.79  of the Philip Ridley play Mercury Fur . It was performed at the Drum Theatre in Plymouth , and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Z[[aza]]
  - -0.61  Robert Boulter , Shane Zaza and Fraser Ayres . " The Guardian noted , " Ben Whishaw and Robert Boulter offer tenderness amid the savagery . "    = = = 2006 –[[ present]]
  - -0.46  was hit three times by bombs that killed 37 men and wounded 52 . Her gunners claimed to have shot down one American dive bomber during the attack . Ise was hit twice during the attack , but her casualties , if[[ any]]
  - -0.45  colleague who was appointed governor general at Chengdu . Despite his financial problems , this was one of the happiest and most peaceful periods of his life . Many of Du 's poems from this period are peaceful depictions of his life at " that[[ched]]
  - -0.44  as noted by Arnell in a MTV News interview .    Shot by November 2012 , the music video was characterised , in several MTV News interviews , as " bigger than anything we 've done before " by Z[[ay]]
Negative pole, set B:
  - -0.89  playing time and production declined .    The Penguins ' needs led them to complete a blockbuster trade on March 1 , 1991 . Cullen was sent to the Hartford Whalers , along with Zarley Z[[al]]
  - -0.55  assisted with the suppression of the mut . Following the restoration of order in the naval base , the armored cruisers Sankt Georg and Kaiser Karl VI were decommissioned and Erzherzog Ferdinand Max and her sisters were stationed in C[[att]]
  - -0.54 @ style mystery cult . Isis and Hermes Trismegistus were both prominent in the Western esoteric tradition that grew from the Roman religious world .    Temples and cults in Egypt itself declined as the Roman economy deteriorated in the third[[ century]]
  - -0.54  the album as " genuinely offensive satanic drivel . " Slayer 's official biography states : " The new sounds disappointed some of the band 's fans who were more accustomed to the style of earlier releases . " Michael Roberts of West[[world]]
  - -0.53  signal and semaphore stations as well as 74 rounds of 190 mm shells aimed at Italian gun @-@ batteries and other port installations .    A major mutiny among crews of the armored cruisers stationed in C[[att]]
Writes up (raises the score of): [' ventric', ' monaster', '"]),', 'amen', 'uclear', ' Bibliography', ' defic', ' Abbas', ' Mahab', '────────', 'ατ', ' tribut']
Writes down (lowers the score of): [' psychology', 'unya', 'udden', 'emporary', 'understand', ' years', '<|endoftext|>', ' last', ' Psychology', 'ijuana', ' len', 'understanding']
Zeroing it: mean loss change -0.0000 nats. Tokens hurt most:
  - +0.30 7 men and wounded 52 . Her gunners claimed to have shot down one American dive bomber during the attack . Ise was hit twice during the attack , but her casualties , if any , are unknown .    The ships were turned[[ into]]
  - +0.20 5 ft 2 in ) wide at its forward end and 13 metres ( 42 ft 8 in ) at the stern . It overhung the stern and increased the overall length of the ships to 219 @[[.]]
  - +0.20  ) more than the preceding class . Their crew consisted of 1 @,@ 360 officers and enlisted men . They had a metacentric height of 1 @.@ 737 metres ( 5 ft [[8]]
  - +0.17  waterline protective belt had a maximum thickness of 299 mm ( 11 @.@ 8 in ) of Vickers cemented armour amidships ; below it was a strake of 100 mm ( [[3]]
  - +0.17 iei at the Naval Battle of Guadalcanal in late 1942 to rudder damage , the IJN decided to reinforce the protection of the steering compartment and to create an auxiliary steering compartment . The protection of the[[ former]]
  - +0.16 acoub . " Kiss You " is an upbeat power pop song with electronic effects ; the lyrics detail a protagonist 's infatuation with a significant other . Critics praised the song for its production , calling it a stand @-@[[ out]]

## O83 (carries part of T3)

Positive pole, set A:
  - +0.53 0 due to digital download sales from its parent album . As a result of an " end @-@ of @-@ year download rush " on the week ending 30 December 2012 , the track re @-[[@]]
  - +0.52  rush " on the week ending 30 December 2012 , the track re @-@ entered the Hot 100 at number 83 . After the accompanying music video was released , the song re @-[[@]]
  - +0.51  the start of the Pacific War , both ships frequently exercised off the coasts of the Soviet Union , Korea and China in addition to training in Japanese waters . Ise hosted Edward , Prince of Wales , and his aide @-@ de @-[[@]]
  - +0.51  ) with the Detroit Lions for one season in 1950 . After retiring from football he settled in Buffalo and became a sports broadcaster . He worked as a color commentator and as a play @-@ by @-[[@]]
  - +0.50  set the stage for every subsequent writer in the genre .    In the 20th century , he was the favourite poet of Kenneth Rexroth , who has described him as " the greatest non @-@ epic , non @-[[@]]
  - +0.50  " a lot of hard work " by Payne , as " pure stupidity " by Louis Tomlinson , and as " I wouldn 't say [ it 's ] comedy , it 's all tongue @-@ in @-[[@]]
  - +0.49  , Du Fu is best known for his lǜshi , a type of poem with strict constraints on form and content , for example :    About two thirds of Du Fu 's 1500 extant works are in this[[ form]]
  - +0.49  praised the song for its production , calling it a stand @-@ out track on Take Me Home .    The track became the group 's sixth top @-@ ten hit in Ireland and the United Kingdom , while attaining top @-[[@]]
Positive pole, set B:
  - +0.61 ptica on a live medley of the two tracks at 1998 's Headbanger 's Heaven festival in the Netherlands . Adrien Begrand of PopMatters described " South of Heaven " as " an un[[orth]]
  - +0.54  aerial advertisement , etc . "    Organisations in the United Kingdom ( UK ) describe GA in less restrictive terms that include elements of commercial aviation . The British Business and General Aviation Association interprets it to be " all aeroplane and helicopter flying[[ except]]
  - +0.52  his illness . The team took an x @-@ ray and found a large black shadow in his chest . He underwent a CAT scan which revealed Cullen had a baseball @-@ sized tumor ; he was diagnosed as having Non @-[[@]]
  - +0.51  . [ Laughs ] I hate the opening riff . It 's what we call a ' happy riff . ' It 's just like ' la @-@ lala @-@ la @-@ la @-[[@]]
  - +0.51  Amun , to form Amun @-@ Ra ; with the solar form of Horus to form Ra @-@ Horakhty ; and with several solar deities as Horemakhet @-@ Khepri @-[[@]]
Negative pole, set A:
  - -1.42  = Background and release = =    " Kiss You " was written by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback , and its producers , Carl Falk and R[[ami]]
  - -1.37  on 7 January 2013 . The song was composed by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback and its producers , Carl Falk and R[[ami]]
  - -1.27 als    Niall Horan — additional guitar    Savan Kotecha — writing , background vocals    Kristian Lundin — writing    Albin Nedler — writing , background vocals    Shellback — writing    R[[ami]]
  - -1.26 ourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . In a review of the production for The Daily Telegraph , theatre critic Charles Spencer noted , " Robert Boulter brings a[[ touching]]
  - -1.23 .@ 71 metres ( 104 ft 0 in ) .    = = Ships = =    = = Service = =    Upon commissioning , the sister ships were assigned to the 1st Battleship Division of the[[ ]]
  - -1.21  room . While under repair at Kure , both ships received prototype Type 21 radars . Commanded by Vice @-@ Admiral Shirō Takasu , the 2nd Battleship Division set sail with the Aleutian Support[[ Group]]
  - -1.17  and of all the intonational potentials of the individual word , qualities that no translation can ever reveal . "    In his lifetime and immediately following his death , Du Fu was not greatly appreciated . In part this can be attributed to his stylistic[[ and]]
  - -1.16  displayed only singly " . He mastered all the forms of Chinese poetry : Chou says that in every form he " either made outstanding advances or contributed outstanding examples " . Furthermore , his poems use a wide range of registers , from the direct[[ and]]
Negative pole, set B:
  - -2.00  could write " slow material as well . " Metal Forces reviewer gives " the band credit for at least making an effort to try something new and not being afraid to experiment at such a crucial stage of their career " , creating " one of the[[ more]]
  - -1.18 o wing in the 1950s fostered the development of hang @-@ gliding during the 1960s and 1970s . The 1960s also saw experiments with motorised[[ hang]]
  - -1.15  creativity , gives form to all things by envisioning and naming them ; Atum produces all things as emanations of himself ; and Amun , according to the myths promoted by his priesthood , preceded and created the other creator gods . These and[[ other]]
  - -1.14  the 6 @.@ 6 centimetres ( 2 @.@ 6 in ) / 45 caliber gun , also manufactured by Škoda . Anti @-@ aircraft and airship protection was covered by the four[[ ]]
  - -1.13  saying “ John Cullen ... beat cancer and came back to play and helped us win . ”    = = Off the ice = =    Cullen and his wife Valerie have three daughters , Kennedy and twins Karlyn and Kort[[land]]
Writes up (raises the score of): ['ikipedia', 'ortunately', 'peutic', 'MetaInfo', 'prudence', 'everal', 'ulumi', 'wargs', 'ailure', 'ceptives', 'ccording', 'queeze']
Writes down (lowers the score of): [' in', '.', '3', '2', '4', '1', '0', 'ot', 'j', ' ', ' natural', '9']
Zeroing it: mean loss change +0.0003 nats. Tokens hurt most:
  - +0.38  a character named " Sean " in Donkey Punch , who tags along with character " Josh " as the " quiet brother ... who hits it off with Tammi " . Boulter guest starred on a two @-@[[ part]]
  - +0.33  Direction performed the track on The Today Show at the Rockefeller Center on 13 November 2012 , to a record crowd estimated at 15 @,@ 000 . " Kiss You " was included in the[[ set]]
  - +0.32  at any angle between -3 and + 20 degrees . In 1921 the elevation was increased to + 30 degrees and then to + 43 degrees during their mid @-@ 1930[[s]]
  - +0.26  on 7 January 2013 . The song was composed by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback and its producers , Carl Falk and R[[ami]]
  - +0.26  were replaced by four geared Kampon turbines with a designed output of 80 @,@ 000 shp ( 60 @,@ 000 kW ) intended to increase their speed to 24 @[[.]]
  - +0.25     Niall Horan — additional guitar    Savan Kotecha — writing , background vocals    Kristian Lundin — writing    Albin Nedler — writing , background vocals    Shellback — writing    Rami[[ Y]]

## O890 (carries part of T3)

Positive pole, set A:
  - +1.16  was not completely suppressed for almost eight years . It caused enormous disruption to Chinese society : the census of 754 recorded 52 @.@ 9 million people , but ten years later , the census counted just 16[[ @]]
  - +1.13  tons ) . This also reduced the draught to 9 @.@ 03 metres ( 29 ft 8 in ) . The overhang of the flight deck at the stern increased the overall length to 219[[ @]]
  - +1.09  weight of the additional armour . In addition , their sterns were lengthened by 7 @.@ 62 metres ( 25 @.@ 0 ft ) . These changes increased their overall length to 213[[ @]]
  - +1.07 95 ft 2 in ) wide at its forward end and 13 metres ( 42 ft 8 in ) at the stern . It overhung the stern and increased the overall length of the ships to 219[[ @]]
  - +1.04  ( 19 ft 8 in ) high and designed to stow nine aircraft . It was fitted with fire fighting foam and carbon dioxide dispensers as a result of wartime experience . The ' T ' -shaped lift was 12[[ @]]
  - +1.03  ) . The overhang of the flight deck at the stern increased the overall length to 219 @.@ 62 metres ( 720 ft 6 in ) and the beam was slightly reduced to 31[[ @]]
  - +0.99  turret weighed 655 long tons ( 666 t ) . The hydraulically powered turrets had an elevation capability of − 5 / + 20 degrees . The guns had a rate of fire of 1[[ @]]
  - +0.96  rotating gunpowder @-@ propelled catapults were fitted on the sides of the hull , forward of the aft superstructure where they partially restricted the arc of fire of the two amidships turrets . They could launch aircraft up to 4[[ @]]
Positive pole, set B:
  - +0.98  interact with those in the sky . The Duat , in contrast , is treated as a remote and inaccessible place , and the gods who dwell there have difficulty communicating with those in the world of the living . The space outside the cosmos is also[[ said]]
  - +0.98  individuals have succeeded in international competition .    Of the 21 @,@ 000 civil aircraft registered in the UK , 96 per cent are engaged in GA operations , and annually the GA fleet accounts for between 1[[ @]]
  - +0.92  farm strips , over 80 per cent of GA activity is conducted at 134 of the larger aerodromes . The GA industry , which is around 7 per cent the size of its CAT cousin , employs 12[[ @]]
  - +0.92  — and demons , who have a more dangerous character . Many demons are hostile , causing illness and other troubles among humans . Their power can also be protective ; they may guard certain places in the Duat , the realm of the dead , or[[ advise]]
  - +0.90  is mostly drawn from religious writings produced by the nation 's scribes and priests . These people were the elite of Egyptian society and were very distinct from the general populace , most of whom were illiterate . Little is known about how well this broader population[[ knew]]
Negative pole, set A:
  - -1.09  ) , as well as in Australia , Canada , Denmark , France , New Zealand , and the Netherlands . The single peaked at number 46 on the US Billboard Hot 100 and has been certified gold by the Recording Industry[[ Association]]
  - -1.08 7 @,@ 000 digital downloads in the US by 18 January 2013 . The single ultimately peaked at number 46 on the Hot 100 and was certified gold by the Recording Industry[[ Association]]
  - -1.03  number 13 on the Australian Singles Chart on 27 January 2013 , marking its peak position and the group 's fourth top twenty hit in Australia . The song has been certified platinum by the Australian Recording Industry[[ Association]]
  - -0.98  It peaked at number 13 in its third and fourth charting weeks , becominh the group 's sixth top @-@ forty appearance in New Zealand . " Kiss You " has received a gold certification from the Recording Industry[[ Association]]
  - -0.94 34 other bombs near missed her , spraying her with splinters and ruptured some hull plates that contaminated some fuel oil and caused leaks in her port boiler rooms . While an exact total of her casualties is not available , it has been estimated[[ that]]
  - -0.90  The An Lushan Rebellion began in December 755 , and was not completely suppressed for almost eight years . It caused enormous disruption to Chinese society : the census of 754 recorded 52 @.@ 9[[ million]]
  - -0.89  its focus to the two remaining rivals for imperial dominance in the Pacific Ocean : Britain and the United States . Satō Tetsutarō , a Japanese Navy admiral and military theorist , speculated that conflict would inevitably arise between Japan and at[[ least]]
  - -0.76  almost eight years . It caused enormous disruption to Chinese society : the census of 754 recorded 52 @.@ 9 million people , but ten years later , the census counted just 16 @.@ 9[[ million]]
Negative pole, set B:
  - -1.55  might be labeled as deities are difficult to count . Egyptian texts list the names of many deities whose nature is unknown and make vague , indirect references to other gods who are not even named . The Egyptologist James P. Allen estimates that more than[[ ]]
  - -1.36 80 per cent of new pilots coming from the GA sector . The CAA estimates that between 1996 and 2006 the number of new professional pilots following the unsponsored training route rose from 48 per[[ cent]]
  - -1.27  such as racing , aerobatics , and parachuting , at which British teams and individuals have succeeded in international competition .    Of the 21 @,@ 000 civil aircraft registered in the UK , 96 per[[ cent]]
  - -1.15  aircraft movements . The pilot training element is regarded by the GA community as a key benefit that is critical to the supply of pilots for the airline industry . It is claimed by the General Aviation Awareness Council that 60 – 70 per[[ cent]]
  - -1.12  pilots coming from the GA sector . The CAA estimates that between 1996 and 2006 the number of new professional pilots following the unsponsored training route rose from 48 per cent to 59 per[[ cent]]
Writes up (raises the score of): ['DateTimeField', 'tidal', 'ebted', 'netes', 'adam', 'eas', 'nces', 'BLE', 'pared', 'itudinal', 'vents', 'rem']
Writes down (lowers the score of): [' per', 'Object', ' thickening', 'ual', 'oub', 'orney', 'terson', ' outl', 'ounce', ' II', 'joice', 'arian']
Zeroing it: mean loss change +0.0001 nats. Tokens hurt most:
  - +0.23 011 film Mercenaries directed by Paris Leonti .    = = Filmography = =    = = = Film = = =    = = = Television = = =    = = = Theatre = = =    = Du[[ Fu]]
  - +0.21  traveled slowly , held up by his ill @-@ health ( by this time he was suffering from poor eyesight , deafness and general old age in addition to his previous ailments ) . They stayed in Kuizhou ( in what is now B[[aid]]
  - +0.19  on Turret No. 5 for a Yokosuka E1Y reconnaissance floatplane in 1928 – 29 . A catapult and a collapsible 4 @- tonne ( 3 @[[.]]
  - +0.16 @ 5 kilograms ( 1 @,@ 485 lb ) and was fired at a muzzle velocity of 770 – 775 metres per second ( 2 @,@ 530 – [[2]]
  - +0.14  a designed output of 80 @,@ 000 shp ( 60 @,@ 000 kW ) intended to increase their speed to 24 @.@ 5 knots ( 45[[ @]]
  - +0.13 @ 0 ft ) . These changes increased their overall length to 213 @.@ 8 metres ( 701 ft ) , their beam to 31 @.@ 75 metres ( 10[[4]]

## O436 (carries part of T3)

Positive pole, set A:
  - +0.93  largely ceremonial . Du Fu 's conscientiousness compelled him to try to make use of it : he caused trouble for himself by protesting the removal of his friend and patron Fang Guan on a petty charge . He was arrested but was p[[ard]]
  - +0.92  of Empress Wu . Du Fu was born in 712 ; the exact birthplace is unknown , except that it was near Luoyang , Henan province ( Gong county is a favourite candidate ) . In later life , he considered[[ himself]]
  - +0.91  of his half brothers . He spent the next four years living in the Luoyang area , fulfilling his duties in domestic affairs .    In the autumn of 744 , he met Li Bai ( Li Po ) for the first[[ time]]
  - +0.79  from the Muromachi period and on scholars and poets in the Edo period , including Matsuo Bashō , the very greatest of all haiku poets . Even in modern Japanese , the term Saint of Poetry ( 詩聖[[ ,]]
  - +0.78  as Hyakuman , Bashō , and Shunkan .    During the Kan 'ei era of the Edo period ( 1624 – 1643 ) , Shào Chuán ( 邵�[[�]]
  - +0.77  two other music videos . The clip depicts the band shooting various scenes via a green screen , which include sequences reminiscent of iconic music videos of songs such as the Beach Boys ' " Surfer Girl " , Elvis Presley 's " J[[ail]]
  - +0.75  and what he hoped for or feared from the progress of various campaigns — these became the enduring themes of his poetry " . Even when he learned of the death of his youngest child , he turned to the suffering of others in his poetry instead of[[ dwelling]]
  - +0.71  it a stand @-@ out track on Take Me Home .    The track became the group 's sixth top @-@ ten hit in Ireland and the United Kingdom , while attaining top @-@ forty positions in both Belgian territories ([[ Flanders]]
Positive pole, set B:
  - +1.26  renamed its Comeback Player of the Year award the John Cullen Award .    Former Lightning head coach Terry Crisp has stated publicly that Cullen was a player that stood out as something special saying “ John Cullen ... beat cancer and came[[ back]]
  - +1.04  and in one game against the Chicago Wolves tied an IHL record when he scored seven points in a 7 – 3 victory .    However , a bout of bronchitis led Cullen to fear that his cancer had returned . Tests came[[ back]]
  - +0.93  Isis poisons the superior god Ra and refuses to cure him unless he reveals his secret name to her . Upon learning the name , she tells it to her son , Horus , and by learning it they gain greater knowledge and power .    In[[ addition]]
  - +0.88  cover version to appear on a Slayer studio album . The song was chosen due to its war @-@ themed lyrics . Hanneman described the track as " more just like one of those odd songs that a lot of people didn 't[[ know]]
  - +0.85 er 's Karl Willetts both rate South of Heaven as amongst the top five albums of all time , while Max Kolesne of Brazilian death metal group Krisiun remembers hearing the song " Silent Scream " for the first[[ time]]
Negative pole, set A:
  - -1.78 okosuka E1Y reconnaissance floatplane in 1928 – 29 . A catapult and a collapsible 4 @- tonne ( 3 @.@ 9 @-@ long @-[[@]]
  - -1.70 5 for a Yokosuka E1Y reconnaissance floatplane in 1928 – 29 . A catapult and a collapsible 4 @- tonne ( 3 @.@ 9 @-[[@]]
  - -1.56  @-@ aircraft defence was provided by four 40 @-@ calibre 3rd Year Type 8 @-@ centimetre AA guns in single mounts . The 7 @.@ 62 @-[[@]]
  - -1.56  the midships magazine and exposed more of the ship to the blast effects of the guns when they fired . Another issue was that Japanese sailors had problems maintaining a high rate of fire with the 45 @.@ 36 @-[[@]]
  - -1.51 931 – 33 the AA guns were replaced with eight 40 @-@ caliber 12 @.@ 7 cm ( 5 @.@ 0 in ) Type 89 dual @-[[@]]
  - -1.50     = = = Propulsion = = =    The Ise @-@ class ships had two sets of direct @-@ drive steam turbines , each of which drove two propeller shafts with 3 @.@ 429 @-[[@]]
  - -1.50  ) to a maximum height of 7 @,@ 500 metres ( 24 @,@ 600 ft ) . The ships were also fitted with six submerged 53 @.@ 3 @-[[@]]
  - -1.50  metres ( 17 @,@ 800 yd ) . Each gun had a rate of fire of up to 10 rounds per minute . Anti @-@ aircraft defence was provided by four 40 @-[[@]]
Negative pole, set B:
  - -1.45  centres , he scored 94 points in the Penguins ' first 65 games and played in his first NHL All @-@ Star Game . However , when Lemieux returned after missing an additional 50 @-[[@]]
  - -1.36 7 in ) torpedo tubes , although rarely used .    = = Service history = =    At the outbreak of World War I , Erzherzog Ferdinand Max was in the III division of the Austrian @-@ Hungarian battle @-[[@]]
  - -1.32  the British 24 cm / 40 ( 9 @.@ 4 " ) Krupp C / 94 , which was used on the Habsburgs . Her secondary armament consisted of twelve 19 @-[[@]]
  - -1.29  Horemakhet @-@ Khepri @-@ Ra @@ Atum . On rare occasion , even deities of different sexes were joined in this way , producing combinations like Osiris @-@ Neith and Mut @-[[@]]
  - -1.29 @ 6 centimetres ( 2 @.@ 6 in ) / 45 caliber gun , also manufactured by Škoda . Anti @-@ aircraft and airship protection was covered by the four 37 @-[[@]]
Writes up (raises the score of): [' Fitzgerald', 'ortunate', ' basket', 'metics', 'ele', ' Fairy', ' freed', ' Movement', 'ULD', ' Walker', ' Extension', ' trough']
Writes down (lowers the score of): ['щ', 'ronomy', 'nec', 'Subthreshold', 'angelo', 'owl', "('.')", 'aku', 'inker', 'AddField', 'reshold', 'nings']
Zeroing it: mean loss change -0.0005 nats. Tokens hurt most:
  - +0.27  as they complicated the protection of the midships magazine and exposed more of the ship to the blast effects of the guns when they fired . Another issue was that Japanese sailors had problems maintaining a high rate of fire with the 45 @.[[@]]
  - +0.24 ualty in 2010 , as " Kieron Fletcher " . Boulter starred in the 2011 film Mercenaries directed by Paris Leonti .    = = Career = =    =[[ =]]
  - +0.22  10 @.@ 4 million views in a 4 @-@ hour period , failing to attain the Vevo record held by Justin Bieber 's " Beauty and a Beat " music video ( 10[[ @]]
  - +0.22  .    = = Charts = =    = = Certifications = =    = = Release history = =    = Ise @-@ class battleship =    The Ise @-@ class battleships ( 伊�[[�]]
  - +0.21 0 mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    = = = Armament = = =    The twelve 45 @-@ calibre 35[[ @]]
  - +0.21  tons ) to 40 @,@ 444 tonnes ( 39 @,@ 805 long tons ) . This also reduced the draught to 9 @.@ 03 metres ( 2[[9]]

(T7: parent pole -, 13 coverage children, children explain 100% of its top tokens)

## T7

Positive pole, set A:
  - +2.41  raised in Kalamazoo , Michigan before his family moved to Saginaw , Michigan . Rifenburg was a star athlete at Saginaw 's Arthur Hill High School in football , basketball , and track and . In 19[[4]]
  - +2.26 5 degree list . The sisters were struck off the Navy List in November and their wrecks were scrapped after the war .    = Dick Rifenburg =    Richard Gale " Dick " Rifenburg ( August 21 , 19[[2]]
  - +2.17  which is a book on high school sports in Saginaw County , Michigan . Rifenburg was named All State in football , basketball and track .    It is ironic that Rifenburg was born in Petoskey , Michigan in 19[[2]]
  - +2.15  of Special Naval Landing Forces to Port Arthur . Three years later , she served as the flagship for the Emperor the puppet state of Manchukuo , Henry Pu @-@ yi , during his state visit to Japan in June 19[[4]]
  - +2.14  , basketball , and track and . In 1943 , Michigan canceled boys high school tournaments in all sports due to World War II , and they did not return until the fall of 1944 . In 19[[4]]
  - +2.14  however , meant that they never actually operated their aircraft in combat . While awaiting their air group the sister ships were sometimes used to ferry troops and material to Japanese bases . They participated in the Battle of Cape Engaño in late 19[[4]]
  - +2.13  Navy List in November and their wrecks were scrapped after the war .    = Dick Rifenburg =    Richard Gale " Dick " Rifenburg ( August 21 , 1926 – December 5 , 19[[9]]
  - +2.12  1994 ) was an American football player and a pioneering television broadcaster for the forerunner to WIVB @-@ TV in Buffalo . He played college football for the University of Michigan Wolverines in 19[[4]]
Positive pole, set B:
  - +2.31  , and on November 20 , 1992 , became Slayer 's second album to be certified gold in the United States . South of Heaven was awarded silver certification in the United on January 1 , 19[[9]]
  - +2.22  chose not to @-@ sign him following the 1993 – 94 season , he returned to the Penguins for one season before Tony Esposito convinced him to sign with the Tampa Bay Lightning in 19[[9]]
  - +2.16  she participated in the bombardment of the Italian port city of Ancona . She also took part in suppressing a major mutiny among the crew members of several armored cruisers stationed in Caro between 1 – 3 February 19[[1]]
  - +2.15  was instead pantheistic , with the power of the sun god present in all other gods , while Hermann Junker argued that Egyptian civilization had been originally monotheistic and became polytheistic in the course of its history .    In 19[[7]]
  - +2.14  returned after missing an additional 50 @-@ games to injury , Cullen 's playing time and production declined .    The Penguins ' needs led them to complete a blockbuster trade on March 1 , 19[[9]]
Negative pole, set A:
  - -3.94  metres per second ( 2 @,@ 640 ft / s ) . A special Type 3 Sanshikidan incendiary shrapnel shell was developed in the 193s for anti @-[[@]]
  - -3.78 9 @,@ 060 mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    = = = Armament = = =    The twelve 45 @-[[@]]
  - -3.77  t ) at deep load . The crew now numbered 1 @,@ 376 officers and enlisted men .    = = = Propulsion = = =    The Ise @-@ class ships had two sets of direct @-[[@]]
  - -3.73 931 – 33 the AA guns were replaced with eight 40 @-@ caliber 12 @.@ 7 cm ( 5 @.@ 0 in ) Type 89 dual @-[[@]]
  - -3.73  t ) each , and eight modern armoured cruisers , 18 @,@ 000 long tons ( 18 @,@ 289 t ) each . This was the genesis of the Eight @-[[@]]
  - -3.72  aunt . He had an elder brother , who died young . He also had three half brothers and one half sister , to whom he frequently refers in his poems , although he never mentions his stepmother .    The son of a scholar @-[[@]]
  - -3.72  adapted from Take Me Home 's liner notes .    = = Charts = =    = = Certifications = =    = = Release history = =    = Ise @-@ class battleship =    The Ise @-[[@]]
  - -3.72 d ) . Each gun had a rate of fire of up to 10 rounds per minute . Anti @-@ aircraft defence was provided by four 40 @-@ calibre 3rd Year Type 8 @-[[@]]
Negative pole, set B:
  - -3.71  his illness . The team took an x @-@ ray and found a large black shadow in his chest . He underwent a CAT scan which revealed Cullen had a baseball @-@ sized tumor ; he was diagnosed as having Non @-[[@]]
  - -3.69  ; and with several solar deities as Horemakhet @-@ Khepri @-@ Ra @@ Atum . On rare occasion , even deities of different sexes were joined in this way , producing combinations like Osiris @-[[@]]
  - -3.67 ome ( province ) of Egypt . Some of these groups contain a specific , symbolically important number of deities . Paired gods can stand for opposite but interrelated concepts that are part of a greater unity . Ra , who is dynamic and light @-[[@]]
  - -3.54 @ 5 knots ( 38 @.@ 0 km / h ; 23 @.@ 6 mph ) .    Erzherzog Ferdinand Max carried a primary armament of four 24 @-[[@]]
  - -3.50  Horemakhet @-@ Khepri @-@ Ra @@ Atum . On rare occasion , even deities of different sexes were joined in this way , producing combinations like Osiris @-@ Neith and Mut @-[[@]]
Writes up (raises the score of): ['-', ' and', '_', ',', '.', 'and', '/', ' A', '\xa0', ' ', ' (', ' d']
Writes down (lowers the score of): [' policym', 'ikipedia', ' burge', ' comprom', '<filename>', 'uthors', 'termedi', 'Intf', ' interspers', ' strugg', ' enric', ' Ishma']
Zeroing it: mean loss change +0.0001 nats. Tokens hurt most:
  - +1.41  Fu 's seemingly negative commentary on the prized horse paintings of Han Gan ignited a controversy that has persisted to the present day .    The tenor of his work changed as he developed his style and adapted to his surroundings ( " chameleon[[ @]]
  - +1.32  000 metres ( 7 @,@ 700 yd ) at 37 knots ( 69 km / h ; 43 mph ) .    In1931 – 33 the[[ AA]]
  - +1.03  a valve in Ise 's No. 2 engine room stuck in the open position and flooded the engine room . While under repair at Kure , both ships received prototype Type 21 radars . Commanded by Vice @-[[@]]
  - +1.03 fer Girl " , Elvis Presley 's " Jailhouse Rock " and Rammstein 's " Mein Land " . The music video received 10 @.@ 4 million views in a 24 @[[-]]
  - +0.94  " , Elvis Presley 's " Blue Hawaii " and Rammstein 's " Mein Land " , among others .    The music video garnered 10 @.@ 4 million views in a 4 @-[[@]]
  - +0.93  was named All State in football , basketball and track .    It is ironic that Rifenburg was born in Petoskey , Michigan in 1926 for several reasons . Ted Petoskey preceded Rifenburg as an All @-[[@]]

## O478 (carries part of T7)

Positive pole, set A:
  - +2.58  too unfathomable . Matsuo Bashō , the greatest haiku poet , was also strongly influenced by Du Fu ; in Oku no Hosomichi , his masterpiece , he cites the first two lines of A Spring View ( �[[�]]
  - +1.51  aunt . He had an elder brother , who died young . He also had three half brothers and one half sister , to whom he frequently refers in his poems , although he never mentions his stepmother .    The son of a scholar @-[[@]]
  - +1.44 9 @,@ 060 mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    = = = Armament = = =    The twelve 45 @-[[@]]
  - +1.40 6 a significant year for himself as well . Achieving All @-@ American status as an end at Michigan would be Rifenburg 's next step after excelling in MHSAA competition .    = = College = =    In the fall[[ of]]
  - +1.40  " , Elvis Presley 's " Blue Hawaii " and Rammstein 's " Mein Land " , among others .    The music video garnered 10 @.@ 4 million views in a 4 @-[[@]]
  - +1.37  early warning radars were added . In September six 30 @-@ round AA rocket launchers were added on the sides of the flight deck .    = = = Flight deck arrangements = = =    A 70 @-[[@]]
  - +1.36  Saginaw 's Arthur Hill High School in football , basketball , and track and . In 1943 , Michigan canceled boys high school tournaments in all sports due to World War II , and they did not return until the fall[[ of]]
  - +1.35  On 15 November the ships were transferred to the 2nd Battleship Division of the 1st Fleet . The sisters were refitted in late 1940 in preparation for war , which included the fitting of external deg[[a]]
Positive pole, set B:
  - +1.91  also been critical of the album in general , describing it as one of his least favorite Slayer albums . He feels vocalist Tom Araya moved too far away from his regular vocal style , and " added too much singing . " Drum[[mer]]
  - +1.41  =    South of Heaven was recorded in Los Angeles , California with Reign in Blood producer Rick Rubin . PopMatters reviewer Adrien Begrand observed that Rubin 's production " shoves [ Dave ] Lombardo 's drum[[ming]]
  - +1.40  material as well . " Metal Forces reviewer gives " the band credit for at least making an effort to try something new and not being afraid to experiment at such a crucial stage of their career " , creating " one of the more original sounding thr[[ash]]
  - +1.38  of pharaonic religion . But many festivals and other traditions of modern Egyptians , both Christian and Muslim , resemble the worship of their ancestors ' gods .    = South of Heaven =    South of Heaven is the fourth studio album by American thr[[ash]]
  - +1.37  , see SMS Erzherzog Ferdinand Max ( 1865 ) .    SMS Erzherzog Ferdinand Max ( German : " His Majesty 's ship Archduke Ferdinand Max " ) was a pre @-[[@]]
Negative pole, set A:
  - -1.21  – 1643 ) , Shào Chuán ( 邵傳 ) of the Ming Dynasty 's Collective Commentary on Du Fu 's Lǜshi ( 杜律集�[[�]]
  - -1.00  Li Bai and Du Fu came to be regarded as representing respectively the Buddhist , Daoist and Confucian strands of Chinese culture . At the same time , the development of Neo @-@ Confucianism ensured that Du Fu , as its poetic exempl[[ar]]
  - -0.89  Road Again Tour ( 2015 )    = = Track listing = =    CD single    " Kiss You " – 3 : 04    " Little Things " – 3 : 42    = = Cred[[its]]
  - -0.89  , or lǜshi ) , and accommodating the complex allusions contained particularly in the later works ( Hawkes writes that " his poems do not as a rule come through very well in translation " — p. ix ) . One[[ extreme]]
  - -0.88  advances or contributed outstanding examples " . Furthermore , his poems use a wide range of registers , from the direct and colloquial to the allusive self @-@ consciously literary . This variety is manifested even within individual works : Owen identifies the , "[[ rapid]]
  - -0.87  according to the digital sheet music published at Musicnotes.com by Sony / ATV Music Publishing . Likewise , Matt Collar from Allmusic noted that the track is " frenetically hyper " . The lyrical content regards the protagonist 's inf[[at]]
  - -0.87  Nedler , Savan Kotecha , Shellback and its producers , Carl Falk and Rami Yacoub . " Kiss You " is an upbeat power pop song with electronic effects ; the lyrics detail a protagonist 's inf[[at]]
  - -0.86  of Du Fu 's achievement , writing in 813 that his predecessor , " united in his work traits which previous men had displayed only singly " . He mastered all the forms of Chinese poetry : Chou says that in every[[ form]]
Negative pole, set B:
  - -2.06  art also depicts places , objects , and concepts in human form . These personified ideas range from deities that were important in myth and ritual to obscure beings , only mentioned once or twice , that may be little more than metaphors .    Confron[[ting]]
  - -0.80  by the Recording Industry Association of America .    In order to offset the pace of the group 's previous album , Slayer deliberately slowed down the album 's tempo . In contrast to their previous , the band utilized undistorted guitars and ton[[ed]]
  - -0.77  . She also took part in suppressing a major mutiny among the crew members of several armored cruisers stationed in Caro between 1 – 3 February 1918 . She also attempted to break through the Otranto Bar[[rage]]
  - -0.73  to contrast the aggressive assault put forth on Reign in Blood , Slayer consciously slowed down the tempo of the album as a whole " , according to Slayer 's official biography . " They also added elements like undistorted guitars and ton[[ed]]
  - -0.72  were decommissioned and Erzherzog Ferdinand Max and her sisters were stationed in Cattaro in their place . On the morning of 11 June , Admiral Miklos Horthy planned a major assault on the Otranto Bar[[rage]]
Writes up (raises the score of): ['uxley', 'anism', 'arman', 'wort', 'affe', 'ching', 'ancock', 'ISTORY', 'entially', 'isexual', 'iken', ' boating']
Writes down (lowers the score of): ['strual', 'phis', '])]', ' Hemisphere', 'ctory', 'aise', 'urethane', 'cumin', ' tar', ' Barriers', ' jour', ' endpoints']
Zeroing it: mean loss change +0.0009 nats. Tokens hurt most:
  - +0.43  16 were operational , respectively . The Japanese plan for the defence of the Philippines envisioned that the surviving carriers would be used to lure the American carrier forces away from the invasion area to a position where the carriers could be attacked by land @[[-]]
  - +0.41  mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    = = = Armament = = =    The twelve 45 @-@ calibre 35 @[[.]]
  - +0.41  at 37 knots ( 69 km / h ; 43 mph ) .    In1931 – 33 the AA guns were replaced with eight 40 @-@ caliber 12 @[[.]]
  - +0.41 5 ft 2 in ) wide at its forward end and 13 metres ( 42 ft 8 in ) at the stern . It overhung the stern and increased the overall length of the ships to 219 @[[.]]
  - +0.41  remaining guns was increased to + 30 degrees .    = = = Protection = = =    The Ise @-@ class ships ' waterline protective belt had a maximum thickness of 299 mm ( 11 @[[.]]
  - +0.40  forces in the winter of 76 , and in the spring of 765 Du Fu and his family sailed down the Yangtze , apparently with the intention of making their way there . They traveled slowly , held up by his ill @[[-]]

(T4: parent pole +, 14 coverage children, children explain 100% of its top tokens)

## T4

Positive pole, set A:
  - +3.64  the rear . It was 6 metres ( 19 ft 8 in ) high and designed to stow nine aircraft . It was fitted with fire fighting foam and carbon dioxide dispensers as a result of wartime experience . The ' T[[ ']]
  - +3.51  forced to move his family due to the turmoil of a famine brought about by massive floods in the region .    In 755 , he received an appointment as Registrar of the Right Commandant 's office of the Crown Prince[[ ']]
  - +3.34 su / Zhejiang area ; his earliest surviving poem , describing a poetry contest , is thought to date from the end of this period , around 735 . In that year , he took the civil service exam , likely in Chang[[ ']]
  - +3.32  712 ; the exact birthplace is unknown , except that it was near Luoyang , Henan province ( Gong county is a favourite candidate ) . In later life , he considered himself to belong to the capital city of Chang[[ ']]
  - +3.30  to various acquaintances . He was relieved by Yan Wu , a friend and former colleague who was appointed governor general at Chengdu . Despite his financial problems , this was one of the happiest and most peaceful periods of his life . Many of Du[[ ']]
  - +3.26  record crowd estimated at 15 @,@ 000 . " Kiss You " was included in the set list of the group 's 3 December 2012 sold @-@ out show at New York City[[ ']]
  - +3.26  abdicate . Du Fu , who had been away from the city , took his family to a place of safety and attempted to join the court of the new emperor ( Suzong ) , but he was captured by the rebels and taken to Chang[[ ']]
  - +3.25  was captured by the rebels and taken to Chang 'an . In the autumn , his youngest son , Du Zongwu ( Baby Bear ) , was born . Around this time Du Fu is thought to have contracted malaria . He escaped from Chang[[ ']]
Positive pole, set B:
  - +3.56  the Penguins won their first Stanley Cup that season , Phil Bourque later said it " broke his heart " that Cullen was not able to share in that championship .    In Hartford , Cullen worked to overcome the team 's fans[[ ']]
  - +3.39  with the " multiplicity of approaches " in Egyptian thought , as well as with the henotheistic practice of ordinary worshippers . He says that the Egyptians may have recognized the unity of the divine by " identifying their uniform notion of ' god[[ ']]
  - +3.37  bonus live disc , and the live double album Decade of Aggression . Lombardo guested with Finnish cellist group Apocalyptica on a live medley of the two tracks at 1998 's Headbanger[[ ']]
  - +3.33  Championship , scoring four points in ten games . Cullen had his best season in 1990 – 91 . As one of the team 's top offensive centres , he scored 94 points in the Penguins[[ ']]
  - +3.31  a National Collegiate Athletic Association East Second Team All @-@ American in 1986 . He graduated as BU 's all @-@ time scoring leader with 241 points , and was named to BU[[ ']]
Negative pole, set A:
  - -1.33  the greatest non @-@ epic , non @-@ dramatic poet who has survived in any language " , and commented that , " he has made me a better man , as a moral agent and as a perceiving organism " .    =[[ =]]
  - -1.32  ships were equipped to operate three floatplanes , although no hangar was provided . The initial Nakajima E4N2 biplanes were replaced by Nakajima E8N2 biplanes in 1938 .    =[[ =]]
  - -1.25  miles ( 14 @,@ 580 km ; 9 @,@ 060 mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    =[[ =]]
  - -1.23  greatest non @-@ epic , non @-@ dramatic poet who has survived in any language " , and commented that , " he has made me a better man , as a moral agent and as a perceiving organism " .    = =[[ =]]
  - -1.22  , France , the Netherlands , and South Korea . In addition , " Kiss You " received gold certifications from the IFPI Norway and Denmark associations , signifying collective shipments of 20 @,@ 000 units .    =[[ =]]
  - -1.20 13 , the song had not been officially promoted to US radio stations . The track , however , was released by Sony Music Entertainment on 8 February 2013 , as the record 's second single in Germany .    =[[ =]]
  - -1.19  due to officials being frequently transferred to the provinces :    = = Works = =    Criticism of Du Fu 's works has focused on his strong sense of history , his moral engagement , and his technical excellence .    = = = History =[[ =]]
  - -1.18 0 in ) rangefinders installed at the top of the pagoda mast at that time . Type 21 air @-@ search radars were installed aboard the sisters in mid @-@ 1942 .    =[[ =]]
Negative pole, set B:
  - -1.28  but had to retreat when the dreadnought SMS Szent István was sunk . After the war , Erzherzog Ferdinand Max was awarded to the United Kingdom as a war prize in 1920 .    =[[ =]]
  - -1.28  was the 1964 introduction of the Learjet 23 . Although it was not the first business jet , it popularised corporate aviation , and established the personal jet as a " whole new class of aircraft " .    =[[ =]]
  - -1.26  praised this musical change , others — more accustomed to the style of earlier releases — were disappointed . The songs " Mandatory Suicide " and the title track , however , have become permanent features of the band 's live setlist .    =[[ =]]
  - -1.26  Hate Crew Deathroll in UK version . Hardcore Punk band , The Mongoloids opened their set with the beginning of " South of Heaven " at Heartfest 7 on May 4 , 2013    =[[ =]]
  - -1.24  than local manifestations of him ; Hathor subsumed a similar cow goddess , Bat ; and an early funerary god , Khenti @-@ Amentiu , was supplanted by Osiris and Anubis .    =[[ =]]
Writes up (raises the score of): [' ', ' (', ',', ' I', '.', ' in', ';', ' as', ' A', ' is', '[', '\xa0']
Writes down (lowers the score of): ['<filename>', ' policym', ' interspers', ' enthus', 'CreateModel', ' psychiat', 'mercial', ' dissemin', ' strugg', ' earthqu', 'firehose', ' dismant']
Zeroing it: mean loss change +0.0052 nats. Tokens hurt most:
  - +2.49  .    The song was included in the dancing game Just Dance 2014 , and is also one of the select songs available on the demo version . Additionally , it is the final main track on the US edition of Now That '[[s]]
  - +2.04  accommodating the complex allusions contained particularly in the later works ( Hawkes writes that " his poems do not as a rule come through very well in translation " — p. ix ) . One extreme on each issue is by Kenneth Rexroth[[ ']]
  - +1.61  2012 , " Kiss You " debuted at number 90 on the United States Billboard Hot 100 due to digital download sales from its parent album . As a result of an " end @-@[[ of]]
  - +1.48  @-@ millimetre ( 6 in ) secondary guns used in the Fusō class and earlier . To resolve this issue , the IJN designed a smaller 140 @-@ millimetre ( 5 @[[.]]
  - +1.47 . 2 in 1927 . It was replaced by a platform on Turret No. 5 for a Yokosuka E1Y reconnaissance floatplane in 1928 – 29 . A catap[[ult]]
  - +1.43  . Despite a 34 % gain in weekly activity to their Vevo channel , with the clip 's success and preceding teaser videos earning 38 million views during the week , One Direction held at number two on the Bill[[board]]

## O1016 (carries part of T4)

Positive pole, set A:
  - +4.19  He had a recurring role in 2003 on two episodes of The Bill , as character " Connor Price " . In 2004 Boulter landed a role as " Craig " in the episode " Teddy[[ ']]
  - +3.92  = = Life = =    Traditional Chinese literary criticism emphasized the life of the author when interpreting a work , a practice which Burton Watson attributes to " the close links that traditional Chinese thought posits between art and morality " . Since many of Du Fu[[ ']]
  - +3.88  and Dominic Hall . Boulter received a favorable review in The Daily Telegraph : " The acting is shatteringly intense , with wired performances from Ben Whishaw ( now unrecognisable from his performance as Trevor Nunn[[ ']]
  - +3.84  that his " famous compassion in fact includes himself , viewed quite objectively and almost as an afterthought " . He therefore " lends grandeur " to the wider picture by comparing it to " his own slightly comical triviality " .    Du Fu[[ ']]
  - +3.80  mere technical restrictions . Hawkes comments that , " it is amazing that Tu Fu is able to use so immensely stylized a form in so natural a manner " .    = = Influence = =    According to the Encyclopædia Britannica , Du Fu[[ ']]
  - +3.76  of vision " .    Although he wrote in all poetic forms , Du Fu is best known for his lǜshi , a type of poem with strict constraints on form and content , for example :    About two thirds of Du Fu[[ ']]
  - +3.72  sufferings of a conscript soldier in the imperial army and a clear @-@ sighted consciousness of suffering . These concerns are continuously articulated in poems on the lives of both soldiers and civilians produced by Du Fu throughout his life .    Although Du Fu[[ ']]
  - +3.69  " Should I learn the poetry of Du Fu and Li Bai ? " Gidō dared to reply , " Yes if you do have enough capability . No if do not . " Since then , there had been many seminars on Du Fu[[ ']]
Positive pole, set B:
  - +4.65  the ironclad present at the Battle of Lissa of the same name , see SMS Erzherzog Ferdinand Max ( 1865 ) .    SMS Erzherzog Ferdinand Max ( German : " His Majesty[[ ']]
  - +4.27  " keep people guessing " as another reason for the musical shift . " In order to contrast the aggressive assault put forth on Reign in Blood , Slayer consciously slowed down the tempo of the album as a whole " , according to Slayer[[ ']]
  - +4.25  Toni Ferguson recorded string quartet adaptations of both tracks on the album The String Quartet Tribute to Slayer : The Evil You Dread , with the former cover being described as having " menacing chord shifts " by AllMusic[[ ']]
  - +4.17  " if you ’ re expecting to hear Reign in Blood Part Two , you ’ ll be in for a major disappointment " .    Kim Neely of Rolling Stone dismissed the album as " genuinely offensive satanic drivel . " Slayer[[ ']]
  - +3.93  has since observed : " There was fire on all the records , but it started dimming when South of Heaven came into the picture . And that 's me personally . Again , I was probably wanting something else . "    Judas Priest[[ ']]
Negative pole, set A:
  - -0.91  @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal[[ Court]]
  - -0.90  due to World War II , and they did not return until the fall of 1944 . In 1944 , he led Arthur Hill High to the MHSAA Class A high school basketball championship ( over Kalam[[az]]
  - -0.88  he portrayed " Scott Parry " in the episode , " Safe Hands " . Boulter starred as " Scott " in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal[[ Court]]
  - -0.87  war due to the MHSAA 's cancellation of state championships in all sports in 1943 .    = = High school = =    Rifenburg was born in Petoskey , Michigan , and raised in Kalam[[az]]
  - -0.85  accidentally collided with and sank a schooner , losing two crewmen . Before the start of the Pacific War , both ships frequently exercised off the coasts of the Soviet Union , Korea and China in addition to training in Japanese waters . I[[se]]
  - -0.83  took on a 5 @-@ degree list that was quickly corrected before she was ordered to tow the crippled carrier Chiyoda to safety . Her attempt was unsuccessful and Chiyoda had to be abandoned to her fate .    I[[se]]
  - -0.83  ( 17 @,@ 930 km ; 11 @,@ 140 mi ) at a speed of 14 knots ( 26 km / h ; 16 mph ) . I[[se]]
  - -0.82  located . This was just the first of a total of five airstrikes that the Americans launched that day . The sisters were not heavily engaged by the early airstrikes which are focusing on the group 's aircraft carriers . I[[se]]
Negative pole, set B:
  - -0.86  were regarded as dependent on divine aid , and after the New Kingdom came to an end , government was increasingly influenced by oracles communicating the gods ' will .    = = = Worship = = =    Official religious practices , which maintained ma[[at]]
  - -0.82  . Yet , according to royal ideology , temple @-@ building was exclusively the pharaoh 's work , as were the rituals that priests usually performed in his stead . These acts were a part of the king 's fundamental role : maintaining ma[[at]]
  - -0.81  definition , suggested by Jan Assmann , says that a deity has a cult , is involved in some aspect of the universe , and is described in mythology or other forms of written tradition . According to a different definition , by Dimitri M[[ee]]
  - -0.78  usually performed in his stead . These acts were a part of the king 's fundamental role : maintaining maat . The king and the nation he represented provided the gods with maat so they could continue to perform their functions , which maintained ma[[at]]
  - -0.78  exclusively the pharaoh 's work , as were the rituals that priests usually performed in his stead . These acts were a part of the king 's fundamental role : maintaining maat . The king and the nation he represented provided the gods with ma[[at]]
Writes up (raises the score of): ['ancial', 'throp', 'asarray', 'ellite', 'alore', 'rapists', 'yad', '?<', 'enties', 'onet', 'netes', '时']
Writes down (lowers the score of): ["']):", "')),", '\'")', ' us', '\n\n ', 'EU', 'ynasty', 'used', ' and', 'rouse', ' both', ' used']
Zeroing it: mean loss change +0.0009 nats. Tokens hurt most:
  - +1.81  although his influence can be seen in some kanshi ( " Chinese poetry made by Japanese poets " ) anthologies such as Bunka Shūreishū in the 9th century . The first notable Japanese appreciator of Du[[ Fu]]
  - +1.49  " Kiss You " proved that the group have an elusive quality . On 12 December 2012 , the group also performed the number on the final of the season of The X Factor USA . Considering One Direction the " franchise[[ ']]
  - +1.41  guns in the forecastle were removed at this time and the maximum elevation of the remaining guns was increased to + 30 degrees .    = = = Protection = = =    The Ise @-@ class ships ' waterline protective[[ belt]]
  - +1.26 in , Albin Nedler , Savan Kotecha , Shellback and its producers , Carl Falk and Rami Yacoub . " Kiss You " is an upbeat power pop song with electronic effects ; the lyrics detail a[[ protagonist]]
  - +1.01 @ class battleships was shaped both by the ongoing international naval arms race and a desire among Japanese naval planners to maintain a fleet of capital ships powerful enough to defeat the United States Navy in an encounter in Japanese territorial waters . The IJN[[ ']]
  - +1.01 14 cm guns in the forecastle were removed at this time and the maximum elevation of the remaining guns was increased to + 30 degrees .    = = = Protection = = =    The Ise @-@ class ships '[[ water]]

(T6: parent pole -, 14 coverage children, children explain 100% of its top tokens)

## T6

Positive pole, set A:
  - +1.69  , the Eight @-@ Eight Fleet Program was restarted .    The first battleships built for the renewed Eight @-@ Eight Fleet Program were the two dreadnoughts of the Kawachi class , ordered in 190[[7]]
  - +1.64  a desire among Japanese naval planners to maintain a fleet of capital ships powerful enough to defeat the United States Navy in an encounter in Japanese territorial waters . The IJN 's fleet of battleships had proven highly successful in 190[[5]]
  - +1.63  his state visit to Japan in June 1940 . On 15 November the ships were transferred to the 2nd Battleship Division of the 1st Fleet . The sisters were refitted in late 194[[0]]
  - +1.58  400 poems in his dense , late style . In autumn 766 , Bo Maolin became governor of the region : he supported Du Fu financially and employed him as his unofficial secretary .    In March 768[[ ,]]
  - +1.58  Cape Engaño in late 1944 , where they decoyed the American carrier fleet supporting the invasion of Leyte away from the landing beaches . Afterwards both ships were transferred to Southeast Asia ; in early 194[[5]]
  - +1.54  was restarted .    The first battleships built for the renewed Eight @-@ Eight Fleet Program were the two dreadnoughts of the Kawachi class , ordered in 1907 and laid down in 190[[8]]
  - +1.47  trained pilots , neither ship ever used its aircraft during combat .    = = = Other changes = = =    After the loss of the fast battleship Hiei at the Naval Battle of Guadalcanal in late 194[[2]]
  - +1.47 ai was already a poetic star . We have twelve poems to or about Li Bai from the younger poet , but only one in the other direction . They met again only once , in 745 .    In 746[[ ,]]
Positive pole, set B:
  - +1.59  pantheistic , with the power of the sun god present in all other gods , while Hermann Junker argued that Egyptian civilization had been originally monotheistic and became polytheistic in the course of its history .    In 1971[[ ,]]
  - +1.56  not to @-@ sign him following the 1993 – 94 season , he returned to the Penguins for one season before Tony Esposito convinced him to sign with the Tampa Bay Lightning in 199[[5]]
  - +1.48  an expanded infrastructure in place , GA became established after the war when manufacturers such as Cessna and Piper introduced light aircraft designed for the private market . The Cessna 172 , developed from the late 194[[0]]
  - +1.45  ships SMS Goeben and SMS Breslau from the British @-@ held Mediterranean ; she advanced as far as Brindisi before being recalled to her home port . Her sole combat engagement occurred in late May 191[[5]]
  - +1.44  Italian gun @-@ batteries and other port installations .    A major mutiny among crews of the armored cruisers stationed in Cattaro , including Sankt Georg and Kaiser Karl VI , began on 1 February 191[[8]]
Negative pole, set A:
  - -3.66  metres per second ( 2 @,@ 640 ft / s ) . A special Type 3 Sanshikidan incendiary shrapnel shell was developed in the 193s for anti @-[[@]]
  - -3.60  the forms of Chinese poetry : Chou says that in every form he " either made outstanding advances or contributed outstanding examples " . Furthermore , his poems use a wide range of registers , from the direct and colloquial to the allusive self @-[[@]]
  - -3.43  adapted from Take Me Home 's liner notes .    = = Charts = =    = = Certifications = =    = = Release history = =    = Ise @-@ class battleship =    The Ise @-[[@]]
  - -3.43 9 @,@ 060 mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    = = = Armament = = =    The twelve 45 @-[[@]]
  - -3.41 931 – 33 the AA guns were replaced with eight 40 @-@ caliber 12 @.@ 7 cm ( 5 @.@ 0 in ) Type 89 dual @-[[@]]
  - -3.38  = =    = = Release history = =    = Ise @-@ class battleship =    The Ise @-@ class battleships ( 伊勢型戦艦 , Ise @-[[@]]
  - -3.30  42 @,@ 675 t ) at deep load . The crew now numbered 1 @,@ 376 officers and enlisted men .    = = = Propulsion = = =    The Ise @-[[@]]
  - -3.29  weight with a higher rate of fire . It also decided that the barbette armour of the earlier ships was too thin and wanted a modest increase in speed to partially counter the higher speeds of the latest foreign ships like the British Queen Elizabeth @-[[@]]
Negative pole, set B:
  - -3.72  ; and with several solar deities as Horemakhet @-@ Khepri @-@ Ra @@ Atum . On rare occasion , even deities of different sexes were joined in this way , producing combinations like Osiris @-[[@]]
  - -3.40  Horemakhet @-@ Khepri @-@ Ra @@ Atum . On rare occasion , even deities of different sexes were joined in this way , producing combinations like Osiris @-@ Neith and Mut @-[[@]]
  - -3.38  permanent ; a god who was involved in one combination continued to appear separately and to form new combinations with other deities . But closely connected deities did sometimes merge . Horus absorbed several falcon gods from various regions , such as Khenty @-[[@]]
  - -3.36  combination continued to appear separately and to form new combinations with other deities . But closely connected deities did sometimes merge . Horus absorbed several falcon gods from various regions , such as Khenty @-@ irty and Khenty @-[[@]]
  - -3.34  his illness . The team took an x @-@ ray and found a large black shadow in his chest . He underwent a CAT scan which revealed Cullen had a baseball @-@ sized tumor ; he was diagnosed as having Non @-[[@]]
Writes up (raises the score of): ['<filename>', ' comprom', 'comings', 'melon', ' Ishma', ',,,,,,,,', ' policym', 'mable', ' pitu', ' pomegran', ' burge', ' strugg']
Writes down (lowers the score of): [' ', ',', '-', '.', ' a', ' I', ' by', ' is', ' as', ' (', '\xa0', ' and']
Zeroing it: mean loss change +0.0040 nats. Tokens hurt most:
  - +1.37 ical triviality " .    Du Fu 's compassion , for himself and for others , was part of his general broadening of the scope of poetry : he devoted many works to topics which had previously been considered unsuitable for poetic treatment . Zhang J[[ie]]
  - +1.09  in Japanese waters in any eventual conflict . Accordingly , the 1907 Imperial Defence Policy called for the construction of a battle fleet of eight modern battleships , 20 @,@ 000 long tons ( 2[[0]]
  - +1.03  high jump 5 feet 8 @.@ 5 inches ( 1 @.@ 74 m ) . He also led Arthur Hill in football , and his high school accomplishments are featured in Glory : The history of Sagin[[aw]]
  - +1.00  at 37 knots ( 69 km / h ; 43 mph ) .    In1931 – 33 the AA guns were replaced with eight 40 @-@ caliber 12 @[[.]]
  - +0.91 926 during gunnery practice . Ise 's AA armament was upgraded in 1931 and Hyūga 's two years later . The latter ship was modernized in 1934 – 3[[6]]
  - +0.85  A @-@ 92 by the IJN and differed enough from the A @-@ 64 design of the Fusō class that it was considered a separate class .    The ships had a length of 20[[8]]

## O478 (carries part of T6)

Positive pole, set A:
  - +2.58  too unfathomable . Matsuo Bashō , the greatest haiku poet , was also strongly influenced by Du Fu ; in Oku no Hosomichi , his masterpiece , he cites the first two lines of A Spring View ( �[[�]]
  - +1.51  aunt . He had an elder brother , who died young . He also had three half brothers and one half sister , to whom he frequently refers in his poems , although he never mentions his stepmother .    The son of a scholar @-[[@]]
  - +1.44 9 @,@ 060 mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    = = = Armament = = =    The twelve 45 @-[[@]]
  - +1.40 6 a significant year for himself as well . Achieving All @-@ American status as an end at Michigan would be Rifenburg 's next step after excelling in MHSAA competition .    = = College = =    In the fall[[ of]]
  - +1.40  " , Elvis Presley 's " Blue Hawaii " and Rammstein 's " Mein Land " , among others .    The music video garnered 10 @.@ 4 million views in a 4 @-[[@]]
  - +1.37  early warning radars were added . In September six 30 @-@ round AA rocket launchers were added on the sides of the flight deck .    = = = Flight deck arrangements = = =    A 70 @-[[@]]
  - +1.36  Saginaw 's Arthur Hill High School in football , basketball , and track and . In 1943 , Michigan canceled boys high school tournaments in all sports due to World War II , and they did not return until the fall[[ of]]
  - +1.35  On 15 November the ships were transferred to the 2nd Battleship Division of the 1st Fleet . The sisters were refitted in late 1940 in preparation for war , which included the fitting of external deg[[a]]
Positive pole, set B:
  - +1.91  also been critical of the album in general , describing it as one of his least favorite Slayer albums . He feels vocalist Tom Araya moved too far away from his regular vocal style , and " added too much singing . " Drum[[mer]]
  - +1.41  =    South of Heaven was recorded in Los Angeles , California with Reign in Blood producer Rick Rubin . PopMatters reviewer Adrien Begrand observed that Rubin 's production " shoves [ Dave ] Lombardo 's drum[[ming]]
  - +1.40  material as well . " Metal Forces reviewer gives " the band credit for at least making an effort to try something new and not being afraid to experiment at such a crucial stage of their career " , creating " one of the more original sounding thr[[ash]]
  - +1.38  of pharaonic religion . But many festivals and other traditions of modern Egyptians , both Christian and Muslim , resemble the worship of their ancestors ' gods .    = South of Heaven =    South of Heaven is the fourth studio album by American thr[[ash]]
  - +1.37  , see SMS Erzherzog Ferdinand Max ( 1865 ) .    SMS Erzherzog Ferdinand Max ( German : " His Majesty 's ship Archduke Ferdinand Max " ) was a pre @-[[@]]
Negative pole, set A:
  - -1.21  – 1643 ) , Shào Chuán ( 邵傳 ) of the Ming Dynasty 's Collective Commentary on Du Fu 's Lǜshi ( 杜律集�[[�]]
  - -1.00  Li Bai and Du Fu came to be regarded as representing respectively the Buddhist , Daoist and Confucian strands of Chinese culture . At the same time , the development of Neo @-@ Confucianism ensured that Du Fu , as its poetic exempl[[ar]]
  - -0.89  Road Again Tour ( 2015 )    = = Track listing = =    CD single    " Kiss You " – 3 : 04    " Little Things " – 3 : 42    = = Cred[[its]]
  - -0.89  , or lǜshi ) , and accommodating the complex allusions contained particularly in the later works ( Hawkes writes that " his poems do not as a rule come through very well in translation " — p. ix ) . One[[ extreme]]
  - -0.88  advances or contributed outstanding examples " . Furthermore , his poems use a wide range of registers , from the direct and colloquial to the allusive self @-@ consciously literary . This variety is manifested even within individual works : Owen identifies the , "[[ rapid]]
  - -0.87  according to the digital sheet music published at Musicnotes.com by Sony / ATV Music Publishing . Likewise , Matt Collar from Allmusic noted that the track is " frenetically hyper " . The lyrical content regards the protagonist 's inf[[at]]
  - -0.87  Nedler , Savan Kotecha , Shellback and its producers , Carl Falk and Rami Yacoub . " Kiss You " is an upbeat power pop song with electronic effects ; the lyrics detail a protagonist 's inf[[at]]
  - -0.86  of Du Fu 's achievement , writing in 813 that his predecessor , " united in his work traits which previous men had displayed only singly " . He mastered all the forms of Chinese poetry : Chou says that in every[[ form]]
Negative pole, set B:
  - -2.06  art also depicts places , objects , and concepts in human form . These personified ideas range from deities that were important in myth and ritual to obscure beings , only mentioned once or twice , that may be little more than metaphors .    Confron[[ting]]
  - -0.80  by the Recording Industry Association of America .    In order to offset the pace of the group 's previous album , Slayer deliberately slowed down the album 's tempo . In contrast to their previous , the band utilized undistorted guitars and ton[[ed]]
  - -0.77  . She also took part in suppressing a major mutiny among the crew members of several armored cruisers stationed in Caro between 1 – 3 February 1918 . She also attempted to break through the Otranto Bar[[rage]]
  - -0.73  to contrast the aggressive assault put forth on Reign in Blood , Slayer consciously slowed down the tempo of the album as a whole " , according to Slayer 's official biography . " They also added elements like undistorted guitars and ton[[ed]]
  - -0.72  were decommissioned and Erzherzog Ferdinand Max and her sisters were stationed in Cattaro in their place . On the morning of 11 June , Admiral Miklos Horthy planned a major assault on the Otranto Bar[[rage]]
Writes up (raises the score of): ['uxley', 'anism', 'arman', 'wort', 'affe', 'ching', 'ancock', 'ISTORY', 'entially', 'isexual', 'iken', ' boating']
Writes down (lowers the score of): ['strual', 'phis', '])]', ' Hemisphere', 'ctory', 'aise', 'urethane', 'cumin', ' tar', ' Barriers', ' jour', ' endpoints']
Zeroing it: mean loss change +0.0009 nats. Tokens hurt most:
  - +0.43  16 were operational , respectively . The Japanese plan for the defence of the Philippines envisioned that the surviving carriers would be used to lure the American carrier forces away from the invasion area to a position where the carriers could be attacked by land @[[-]]
  - +0.41  mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    = = = Armament = = =    The twelve 45 @-@ calibre 35 @[[.]]
  - +0.41  at 37 knots ( 69 km / h ; 43 mph ) .    In1931 – 33 the AA guns were replaced with eight 40 @-@ caliber 12 @[[.]]
  - +0.41 5 ft 2 in ) wide at its forward end and 13 metres ( 42 ft 8 in ) at the stern . It overhung the stern and increased the overall length of the ships to 219 @[[.]]
  - +0.41  remaining guns was increased to + 30 degrees .    = = = Protection = = =    The Ise @-@ class ships ' waterline protective belt had a maximum thickness of 299 mm ( 11 @[[.]]
  - +0.40  forces in the winter of 76 , and in the spring of 765 Du Fu and his family sailed down the Yangtze , apparently with the intention of making their way there . They traveled slowly , held up by his ill @[[-]]

(T8: parent pole -, 21 coverage children, children explain 100% of its top tokens)

## T8

Positive pole, set A:
  - +3.63  as noted by Arnell in a MTV News interview .    Shot by November 2012 , the music video was characterised , in several MTV News interviews , as " bigger than anything we 've done before " by Z[[ay]]
  - +1.52 923 Great Kantō earthquake struck , they sailed to Kyushu where they loaded supplies from for the victims on 4 September . Together with two other battleships and a pair of light cruisers , Ise sank the destroyer Y[[ay]]
  - +1.29  the group 's manager , had challenged prominent songwriters to compete for space on One Direction 's second album . Falk said , " It 's important to get their personalities on the music . " In addition , the article reported that Sy[[co]]
  - +1.07  . These concerns are continuously articulated in poems on the lives of both soldiers and civilians produced by Du Fu throughout his life .    Although Du Fu 's frequent references to his own difficulties can give the impression of an all @-@ consuming sol[[ips]]
  - +0.93  9 @,@ 060 mi ) a speed of 16 knots ( 30 km / h ; 18 mph ) .    = = = Armament = = =    The twelve 45 @[[-]]
  - +0.91  of the conning tower were 305 mm thick .    The Ise class were the only Japanese battleships to place the powder magazine above the shell magazine as the IJN wished to put as much space as possible between the[[ highly]]
  - +0.88  rejected on the grounds of expense and , most critically , time , and the IJN settled on removing the rear pair of turrets and replacing them with a flight deck equipped with two catapults to launch floatplanes . The Ise @[[-]]
  - +0.87  writing , production , programming , instruments , bass    Credits adapted from Take Me Home 's liner notes .    = = Charts = =    = = Certifications = =    = = Release history = =    = Ise @[[-]]
Positive pole, set B:
  - +2.19  establishment of the independent CAA in 1972 .    With an expanded infrastructure in place , GA became established after the war when manufacturers such as Cessna and Piper introduced light aircraft designed for the private market . The C[[ess]]
  - +1.78  aviation in the UK , and this task remained the responsibility of government departments until the establishment of the independent CAA in 1972 .    With an expanded infrastructure in place , GA became established after the war when manufacturers such as C[[ess]]
  - +1.64  in place , GA became established after the war when manufacturers such as Cessna and Piper introduced light aircraft designed for the private market . The Cessna 172 , developed from the late 1940s C[[ess]]
  - +1.62  The introduction of the de Havilland DH.60 Moth in 1925 revolutionised light aviation , and the Royal Aero Club , recognising the " vital necessity of promoting civil flying " , formed the Light Aeropl[[ane]]
  - +1.41  search and rescue , aerial advertisement , etc . "    Organisations in the United Kingdom ( UK ) describe GA in less restrictive terms that include elements of commercial aviation . The British Business and General Aviation Association interprets it to be " all aeropl[[ane]]
Negative pole, set A:
  - -5.66  104 barrels , and a pair of Type 13 early warning radars were added . In September six 30 @-@ round AA rocket launchers were added on the sides of the flight deck .    = =[[ =]]
  - -5.49  the fifteen @-@ round magazines . In addition the forward pair of 14 cm guns in the forecastle were removed at this time and the maximum elevation of the remaining guns was increased to + 30 degrees .    = =[[ =]]
  - -5.48  Matsuda Chiaki 's reformed Fourth Carrier Division of the 3rd Fleet . The division 's 634th Naval Air Group was formed that same day and conducted its first catapult launches in late June .    = =[[ =]]
  - -5.48  ships were equipped to operate three floatplanes , although no hangar was provided . The initial Nakajima E4N2 biplanes were replaced by Nakajima E8N2 biplanes in 1938 .    =[[ =]]
  - -5.47  140 mm . Inside the original skin of the ships , two torpedo bulkheads were also added and the turret roofs were increased to a total of 152 millimetres ( 6 in ) of armour .    = =[[ =]]
  - -5.47  6 could not elevate to the full + 43 degrees deemed necessary for the long @-@ range engagement anticipated by the IJN . The Fusōs were scheduled to follow once the first two were completed .    = =[[ =]]
  - -5.41  Prince 's Palace . Although this was a minor post , in normal times it would have been at least the start of an official career . Even before he had begun work , however , the position was swept away by events .    = =[[ =]]
  - -5.39  " . In 762 , he left the city to escape a rebellion , but he returned in summer 764 when he was appointed an advisor to Yan , who was involved in campaigns against the Tibetan Empire .    = =[[ =]]
Negative pole, set B:
  - -5.61 97 ; Cullen was leading the team in scoring , but was suffering flu @-@ like symptoms that he could not shake . As Tampa was fighting for a playoff spot , he played through his condition for weeks .    = =[[ =]]
  - -5.60  Hate Crew Deathroll in UK version . Hardcore Punk band , The Mongoloids opened their set with the beginning of " South of Heaven " at Heartfest 7 on May 4 , 2013    =[[ =]]
  - -5.58  numbers moving " at the sludgier of Black Sabbath . " Araya commented that the " album was a late bloomer — it wasn 't really received well , but it kind of grew on everybody later . "    =[[ =]]
  - -5.57  felt it made Slayer seem as though they " had matured a little bit " , while Friedman himself deemed it " a really cool back cover " and " one of the most classic shots of them [ Slayer ] ever . "    =[[ =]]
  - -5.52  Erzherzog Karl . Erzherzog Ferdinand Max was also fitted with two above water 45 @-@ centimeter ( 17 @.@ 7 in ) torpedo tubes , although rarely used .    =[[ =]]
Writes up (raises the score of): ['<filename>', ' strugg', ' dissemin', ' enthus', ' affili', ' comprom', ' authent', ' psychiat', ' mortg', ').|', ' necessit', ' interpre']
Writes down (lowers the score of): ['.', '-', ' to', ',', '\xa0', ' of', '\n', '/', ' "', ':', ' or', ' (']
Zeroing it: mean loss change +0.0029 nats. Tokens hurt most:
  - +2.11  Hung believes that frustration is a more likely reason . He next spent around six weeks in Qinzhou ( now Tianshui , Gansu province ) , where he wrote more than sixty poems .    = = = Chengdu = =[[ =]]
  - +1.72  . Although this was a minor post , in normal times it would have been at least the start of an official career . Even before he had begun work , however , the position was swept away by events .    = = = War = =[[ =]]
  - +1.31  Nunn 's Hamlet ) , Robert Boulter , Shane Zaza and Fraser Ayres . " The Guardian noted , " Ben Whishaw and Robert Boulter offer tenderness amid the savagery . "    = =[[ =]]
  - +1.24 @ round magazines . In addition the forward pair of 14 cm guns in the forecastle were removed at this time and the maximum elevation of the remaining guns was increased to + 30 degrees .    = = = Protection = =[[ =]]
  - +1.17 2 @,@ 001 long tons ( 42 @,@ 675 t ) at deep load . The crew now numbered 1 @,@ 376 officers and enlisted men .    = =[[ =]]
  - +1.16  wording and themes . It is said that when he died in Osaka during a long travel , a copy of Du Fu 's poetry was found with him as one of a few precious items which he was able to carry around .    =[[ =]]

## O381 (carries part of T8)

Positive pole, set A:
  - +1.40  excellence = = =    Du Fu 's work is notable above all for its range . Chinese critics traditionally used the term 集大成 ( jídàchéng- " complete symphony " ) , a reference to Men[[cius]]
  - +1.37 8 @.@ 5 inches ( 1 @.@ 74 m ) . He also led Arthur Hill in football , and his high school accomplishments are featured in Glory : The history of Saginaw County sports by Jack T[[any]]
  - +0.88  that Du Fu 's reputation reached its peak . In this period a comprehensive re @-@ evaluation of earlier poets took place , in which Wang Wei , Li Bai and Du Fu came to be regarded as representing respectively the Buddhist , D[[ao]]
  - +0.82 4 July in preparation . They remained on " standby alert " until the actual conversions began . Ise was converted at Kure Naval Arsenal from 23 February to 5 September 1943 and Hyūga at S[[ase]]
  - +0.72  " Live While We 're Young " . In April 2012 , The Independent reported that Simon Cowell , the group 's manager , had challenged prominent songwriters to compete for space on One Direction 's second album .[[ Falk]]
  - +0.68  =    " Kiss You " was written by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback , and its producers , Carl Falk and Rami Yacoub .[[ Falk]]
  - +0.59  third overall single on 7 January 2013 . The song was composed by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback and its producers , Carl[[ Falk]]
  - +0.59  ( from around 750 ) , gives voice to the sufferings of a conscript soldier in the imperial army and a clear @-@ sighted consciousness of suffering . These concerns are continuously articulated in poems on the lives of both soldiers and[[ civilians]]
Positive pole, set B:
  - +1.08 er and the harvest goddess Renenutet — and demons , who have a more dangerous character . Many demons are hostile , causing illness and other troubles among humans . Their power can also be protective ; they may guard certain places in the Du[[at]]
  - +1.06  instance where a god dies is the myth of Osiris ' murder , in which that god is resurrected as ruler of the Duat . The sun god is also said to grow old during his daily journey across the sky , sink into the Du[[at]]
  - +1.05  were said to live in those places instead . Most events of mythology , set in a time before the gods ' withdrawal from the human realm , take place in an earthly setting . The deities there sometimes interact with those in the sky . The Du[[at]]
  - +1.04  the sky , sink into the Duat at night , and emerge as a young child at dawn . In the process he comes into contact with the rejuvenating water of primordial chaos . Funerary texts that depict Ra 's journey through the Du[[at]]
  - +1.04  night , continue this struggle into the present . Another prominent theme is the gods ' death and revival . The clearest instance where a god dies is the myth of Osiris ' murder , in which that god is resurrected as ruler of the Du[[at]]
Negative pole, set A:
  - -4.44  colleague who was appointed governor general at Chengdu . Despite his financial problems , this was one of the happiest and most peaceful periods of his life . Many of Du 's poems from this period are peaceful depictions of his life at " that[[ched]]
  - -4.25  Li Bai on aesthetic grounds from attacks made against them . Both these writers showed the influence of Du Fu in their own poetic work . By the beginning of the 10th century , Wei Zhuang constructed the first replica of his that[[ched]]
  - -2.24  of 104 barrels , and a pair of Type 13 early warning radars were added . In September six 30 @-@ round AA rocket launchers were added on the sides of the flight deck .    =[[ =]]
  - -2.22  Crown Prince 's Palace . Although this was a minor post , in normal times it would have been at least the start of an official career . Even before he had begun work , however , the position was swept away by events .    =[[ =]]
  - -2.22  " while Chris Younie , a critic from 4Music , deemed it an " amazing pop song " , lauding the group 's falsetto and its " head @-@ banging anthemic " chorus .    =[[ =]]
  - -2.18  the greatest non @-@ epic , non @-@ dramatic poet who has survived in any language " , and commented that , " he has made me a better man , as a moral agent and as a perceiving organism " .    =[[ =]]
  - -2.15  wording and themes . It is said that when he died in Osaka during a long travel , a copy of Du Fu 's poetry was found with him as one of a few precious items which he was able to carry around .    =[[ =]]
  - -2.13  ships were equipped to operate three floatplanes , although no hangar was provided . The initial Nakajima E4N2 biplanes were replaced by Nakajima E8N2 biplanes in 1938 .    =[[ =]]
Negative pole, set B:
  - -3.08  events . Some deities who were closely connected with isfet , such as Set , could be blamed for disorder within the world without placing guilt on the other gods . But some writings do accuse the deities of causing human misery , while others give the[[odic]]
  - -2.65  of phenomena they represented include physical places and objects as well as abstract concepts and forces . The god Shu was the deification of all the world 's air ; the goddess Meretseger oversaw a limited region of the earth , the The[[ban]]
  - -2.53  the temple building . Communities also built and managed small chapels for their own use , and some families had shrines inside their homes . Despite the gulf that separated humanity from the divine , the Egyptians were by opportunities to approach their gods .    =[[ =]]
  - -2.46  temple building . Communities also built and managed small chapels for their own use , and some families had shrines inside their homes . Despite the gulf that separated humanity from the divine , the Egyptians were by opportunities to approach their gods .    = =[[ =]]
  - -2.37  praised this musical change , others — more accustomed to the style of earlier releases — were disappointed . The songs " Mandatory Suicide " and the title track , however , have become permanent features of the band 's live setlist .    =[[ =]]
Writes up (raises the score of): [' acres', ' footing', ' fluxes', ' captions', '[:,:,', ' distingu', ' enthus', ' shutil', 'Ops', 'ctic', ' stomachs', ' sexism']
Writes down (lowers the score of): ['rapers', 'rowsiness', 'rror', 'operability', 'ABASE', 'enaissance', ' either', 'idisciplinary', 'setObject', 'angerous', ' backward', ' range']
Zeroing it: mean loss change +0.0010 nats. Tokens hurt most:
  - +0.86  colleague who was appointed governor general at Chengdu . Despite his financial problems , this was one of the happiest and most peaceful periods of his life . Many of Du 's poems from this period are peaceful depictions of his life at " that[[ched]]
  - +0.73  the viewer should have a hard time disliking the group . Mikael Wood , the critic for Los Angeles Times , commended the group for " having a genuinely great time " , rather than going through the motions .    = = Live[[ performances]]
  - +0.63  ( Allied reporting name " Paul " ) Both aircraft had development problems and neither air group ever had all of its intended aircraft . Coupled with a shortage of trained pilots , neither ship ever used its aircraft during combat .    = = = Other[[ changes]]
  - +0.60  look like you know what you 're doing . " Boulter starred in the 2011 film Mercenaries directed by Paris Leonti .    = = Filmography = =    = = = Film = =[[ =]]
  - +0.55  , and is also one of the select songs available on the demo version . Additionally , it is the final main track on the US edition of Now That 's What I Call Music ! 46 .    = = Background and release =[[ =]]
  - +0.54  . It is said that when he died in Osaka during a long travel , a copy of Du Fu 's poetry was found with him as one of a few precious items which he was able to carry around .    = = Translation =[[ =]]

(T5: parent pole +, 9 coverage children, children explain 97% of its top tokens)

## T5

Positive pole, set A:
  - +2.99  as Hyakuman , Bashō , and Shunkan .    During the Kan 'ei era of the Edo period ( 1624 – 1643 ) , Shào Chuán ( 邵�[[�]]
  - +2.46  result of wartime experience . The ' T ' -shaped lift was 12 @@ 1 metres ( 39 ft 8 in ) wide at its forward end and 6 @.@ 6 metres ( 21[[ ft]]
  - +2.46 0 @-@ metre @-@ long ( 229 ft 8 in ) flight deck was built above the stern and stretched forward to the rebuilt aft superstructure . The flight deck was 29 metres ( 95[[ ft]]
  - +2.43  high and designed to stow nine aircraft . It was fitted with fire fighting foam and carbon dioxide dispensers as a result of wartime experience . The ' T ' -shaped lift was 12 @@ 1 metres ( 39[[ ft]]
  - +2.42  flight deck was built above the stern and stretched forward to the rebuilt aft superstructure . The flight deck was 29 metres ( 95 ft 2 in ) wide at its forward end and 13 metres ( 42[[ ft]]
  - +2.42 3 ft 0 in ) overall , a beam of 28 @.@ 65 metres ( 94 ft 0 in ) and a draught of 8 @.@ 93 metres ( 29[[ ft]]
  - +2.42  and 13 metres ( 42 ft 8 in ) at the stern . It overhung the stern and increased the overall length of the ships to 219 @.@ 62 metres ( 720[[ ft]]
  - +2.41  more in temporary positions on the flight deck for a total of thirteen .    The 40 @-@ metre @-@ long ( 131 ft 3 in ) hangar was 20 metres ( 65[[ ft]]
Positive pole, set B:
  - +2.37 4 @-@ centimeter ( 9 @.@ 4 in ) / 40 caliber guns in two twin turrets on the centerline . These guns were an Austro @-@ Hungarian replica of the British 24[[ cm]]
  - +2.23 20 @,@ 00 metres ( 22 @,@ 000 yd ) at maximum elevation with a muzzle velocity of 800 metres per second ( 2 @,@ 600[[ ft]]
  - +2.16  ship and two twin turrets on the centerline. shell 20 @,@ 00 metres ( 22 @,@ 000 yd ) at maximum elevation with a muzzle velocity of 800[[ metres]]
  - +2.07  Austro Hungarian navy , Erzherzog Ferdinand Max took a minor part in the bombardment of Ancona on 24 May 1915 . There she and her sisters expended 24 rounds of 240[[ mm]]
  - +2.06  May 1915 . There she and her sisters expended 24 rounds of 240 mm armor @-@ piercing shells at signal and semaphore stations as well as 74 rounds of 190[[ mm]]
Negative pole, set A:
  - -3.89  and Dominic Hall . Boulter received a favorable review in The Daily Telegraph : " The acting is shatteringly intense , with wired performances from Ben Whishaw ( now unrecognisable from his performance as Trevor Nunn[[ ']]
  - -3.87  his influence as that of Shakespeare in England : it was hard for any Chinese poet not to be influenced by him . While there was never another Du Fu , individual poets followed in the traditions of specific aspects of his work : Bai Juyi[[ ']]
  - -3.83  at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as " Craig " in the episode " Teddy[[ ']]
  - -3.81  it was hard for any Chinese poet not to be influenced by him . While there was never another Du Fu , individual poets followed in the traditions of specific aspects of his work : Bai Juyi 's concern for the poor , Lu You[[ ']]
  - -3.77  He had a recurring role in 2003 on two episodes of The Bill , as character " Connor Price " . In 2004 Boulter landed a role as " Craig " in the episode " Teddy[[ ']]
  - -3.76  . Since the establishment of the People 's Republic of China , Du Fu 's loyalty to the state and concern for the poor have been interpreted as embryonic nationalism and socialism , and he has been praised for his use of simple , " people[[ ']]
  - -3.75  that his " famous compassion in fact includes himself , viewed quite objectively and almost as an afterthought " . He therefore " lends grandeur " to the wider picture by comparing it to " his own slightly comical triviality " .    Du Fu[[ ']]
  - -3.73  interviews , as " bigger than anything we 've done before " by Zayn Malik , as " a lot of hard work " by Payne , as " pure stupidity " by Louis Tomlinson , and as " I wouldn[[ ']]
Negative pole, set B:
  - -3.87  the ironclad present at the Battle of Lissa of the same name , see SMS Erzherzog Ferdinand Max ( 1865 ) .    SMS Erzherzog Ferdinand Max ( German : " His Majesty[[ ']]
  - -3.81  ' representative and managed the temples where the rituals were carried out .    The gods ' complex characteristics were expressed in myths and in intricate relationships between deities : family ties , loose groups and hierarchies , and combinations of separate gods into one . Deities[[ ']]
  - -3.80  with the " multiplicity of approaches " in Egyptian thought , as well as with the henotheistic practice of ordinary worshippers . He says that the Egyptians may have recognized the unity of the divine by " identifying their uniform notion of ' god[[ ']]
  - -3.73  particular deities , including their height and eye color . Yet these characteristics are not fixed ; in myths , gods change their appearances to suit their own purposes . Egyptian texts often refer to deities ' true , underlying forms as " mysterious " . The Egyptians[[ ']]
  - -3.72  as part of its recovery plan in response to the Automotive industry crisis of 2008 – 2010 . He has since returned to his brother 's dealership , serving as its general manager .    Cullen[[ ']]
Writes up (raises the score of): [' and', ' (', ',', ' in', ' ', ' is', ' of', '-', ' as', ' or', ' "', '.']
Writes down (lowers the score of): ['<filename>', ' strugg', ' dissemin', ' comprom', ' affili', ' enthus', ' auton', ' necessit', ' interspers', ' interpre', ' dismant', ' accred']
Zeroing it: mean loss change +0.0013 nats. Tokens hurt most:
  - +1.54 . 2 in 1927 . It was replaced by a platform on Turret No. 5 for a Yokosuka E1Y reconnaissance floatplane in 1928 – 29 . A catap[[ult]]
  - +1.04  issue , the IJN designed a smaller 140 @-@ millimetre ( 5 @.@ 5 in ) gun that offset its lighter shell weight with a higher rate of fire . It also decided that the[[ barb]]
  - +0.81  killed and some 111 – 121 crewmen were wounded during this attack . Hyūga was unsuccessfully attacked by an American submarine at 18 : 43 . Around 19 : 00[[ Oz]]
  - +0.80  35 @,@ 450 meters ( 38 @,@ 770 yd ) at + 43 degrees after modernization . Also available was a 625 @-@ kilogram ( [[1]]
  - +0.79  magazines was increased to a total thickness of 140 mm . Inside the original skin of the ships , two torpedo bulkheads were also added and the turret roofs were increased to a total of 152 millimetres ( 6[[ in]]
  - +0.73  0 @.@ 98 in ) guns had an effective range of 1 @,@ 500 – 3 @,@ 000 metres ( 1 @,@ 600 – [[3]]

## O886 (carries part of T5)

Positive pole, set A:
  - +1.30 7 January 2013 . The song was composed by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback and its producers , Carl Falk and Rami Y[[ac]]
  - +1.27  and release = =    " Kiss You " was written by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback , and its producers , Carl Falk and Rami Y[[ac]]
  - +1.21  by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback , and its producers , Carl Falk and Rami Yacoub . Falk , Kotecha , and Y[[ac]]
  - +1.16  much better shape and the Japanese decided to retain the aircraft ashore for use against the American carriers . The Fourth Carrier Division was assigned to the Northern Force under the command of Vice Admiral Jisaburō Ozawa and the sisters sailed from Y[[ash]]
  - +1.09  Niall Horan — additional guitar    Savan Kotecha — writing , background vocals    Kristian Lundin — writing    Albin Nedler — writing , background vocals    Shellback — writing    Rami Y[[ac]]
  - +0.96  @,@ 100 yd ) ; they had a ceiling of 9 @,@ 440 metres ( 30 @,@ 970 ft ) at their maximum elevation of + 90[[ degrees]]
  - +0.94  @-@ de @-@ camp Lieutenant Louis Mountbatten in 1922 during the prince 's visit to Japan . In Korea Bay when the 1923 Great Kantō earthquake struck , they sailed to Ky[[ush]]
  - +0.90  of fire was around eight rounds per minute . Two twin @-@ gun mounts for license @-@ built Vickers two @-@ pounder light AA guns were also added . These guns had a maximum elevation of + 80[[ degrees]]
Positive pole, set B:
  - +1.26  velocity of 800 metres per second ( 2 @,@ 600 ft / s ) . The gun weighed 12 @.@ 1 tons and could fire three rounds per minute.The ships had a[[ tertiary]]
  - +1.13  fitting dress for goddesses .    The basic anthropomorphic form varies . Child gods are depicted nude , as are some adult gods when their procreative powers are emphasized . Certain male deities are given heavy bellies and breasts , signifying either and[[rog]]
  - +0.73  a " whole new class of aircraft " .    = = Activities = =    The GA sector operates a range of aircraft , including balloons and airships , gliders , hang gliders , paragliders , microlights , gy[[roc]]
  - +0.69 ardo remembers listening to a live rendition of " South of Heaven " and thinking " ‘ Man ! There 's just so much groove in that song . ’ To my kids I was saying , ‘ Listen to that ! Listen to how gro[[ov]]
  - +0.58  groove in that song . ’ To my kids I was saying , ‘ Listen to that ! Listen to how groovy that is ! ’ And it 's heavy . " A rare live version of the track featured on the JÄ[[GER]]
Negative pole, set A:
  - -1.69  sports shows and was eventually inducted into the Buffalo Broadcasters Hall of Fame .    In college , he led the Big Ten Conference in single season receptions during his senior year and set Michigan Wolverines receptions records for both career touch[[down]]
  - -1.32 azoo Central High School ) , scoring 24 points , including 17 in the second half , of the championship game . Rifenburg was also the state champion in 1944 in both the shot put 46[[ feet]]
  - -1.30 ial son , an affectionate father , a generous brother , a faithful husband , a loyal friend , a dutiful official , and a patriotic subject . "    Below is an example of one of Du Fu 's later works , To My Ret[[ired]]
  - -1.24  the Buffalo Broadcasters Hall of Fame .    In college , he led the Big Ten Conference in single season receptions during his senior year and set Michigan Wolverines receptions records for both career touchdown and single @-@ season touch[[downs]]
  - -1.06  their tripod foremasts . Both ships were also given torpedoges to improve their underwater protection and to compensate for the weight of the additional armour . In addition , their sterns were lengthened by 7 @.@ 62[[ metres]]
  - -1.04  high and designed to stow nine aircraft . It was fitted with fire fighting foam and carbon dioxide dispensers as a result of wartime experience . The ' T ' -shaped lift was 12 @@ 1 metres ( 39[[ ft]]
  - -1.04 9 dual @-@ purpose guns , fitted on both sides of the forward superstructures in four twin @-@ gun mounts . When firing at surface targets , the guns had a range of 14 @,@ 700[[ metres]]
  - -1.04  , capped shells . Each of these shells weighed 673 @.@ 5 kilograms ( 1 @,@ 485 lb ) and was fired at a muzzle velocity of 770 – 775[[ metres]]
Negative pole, set B:
  - -1.11 4 feet 2 inches ( 126 @.@ 2 m ) long , had a beam of 71 feet 5 inches ( 21 @.@ 8 m ) and a draft of 24[[ feet]]
  - -1.11 2 long tons ( 10 @,@ 640 t ) . She was 414 feet 2 inches ( 126 @.@ 2 m ) long , had a beam of 71[[ feet]]
  - -0.97  / 42 caliber guns , also made by Škoda , mounted in eight single casemates on either wing of the ship and two twin turrets on the centerline. shell 20 @,@ 00[[ metres]]
  - -0.90  .    = = Design = =    Erzherzog Ferdinand Max displaced 10 @,@ 472 long tons ( 10 @,@ 640 t ) . She was 414[[ feet]]
  - -0.88  playing time and production declined .    The Penguins ' needs led them to complete a blockbuster trade on March 1 , 1991 . Cullen was sent to the Hartford Whalers , along with Zarley Z[[al]]
Writes up (raises the score of): ['iou', 'TYPES', ' certific', 'IU', ' utilise', 'aiser', 'anni', 'BaseModel', ' Yours', "']):", ' vermic', ' evid']
Writes down (lowers the score of): [' <', 'allclose', ' like', 'elect', 'rail', 'verages', 'h', 'abs', ' voltage', 'aint', ' acid', 'voltage']
Zeroing it: mean loss change +0.0000 nats. Tokens hurt most:
  - +0.26 masts . Both ships were also given torpedoges to improve their underwater protection and to compensate for the weight of the additional armour . In addition , their sterns were lengthened by 7 @.@ 62 metres ( 2[[5]]
  - +0.17  The song was included in the dancing game Just Dance 2014 , and is also one of the select songs available on the demo version . Additionally , it is the final main track on the US edition of Now That 's What[[ I]]
  - +0.14  Spy noted the song 's possibility to become an international hit , applauding it sonically . A reviewer for MTV News described the track 's lyricism as " butterflies @-@ inducing " , and Sam Lansky of Id[[ol]]
  - +0.14  career was interrupted by World War II service , and his high school career was also affected by the war due to the MHSAA 's cancellation of state championships in all sports in 1943 .    = = High school[[ =]]
  - +0.14  fitted at that time and data from the rangefinders had to be processed manually . Turrets 2 , 3 , and 5 were built with imported 6 @-@ metre ( 19 ft 8 in ) Ba[[usch]]
  - +0.13     A 70 @-@ metre @-@ long ( 229 ft 8 in ) flight deck was built above the stern and stretched forward to the rebuilt aft superstructure . The flight deck was 29 metres[[ (]]

(T9: parent pole +, 13 coverage children, children explain 62% of its top tokens)

## T9

Positive pole, set A:
  - +5.78  manner " .    = = Influence = =    According to the Encyclopædia Britannica , Du Fu 's writings are considered by many literary critics to be among the greatest of all time , and it states " his dense , compressed language makes use[[ of]]
  - +5.71 an the following year , and was appointed Reminder when he rejoined the court in May 757 . This post gave access to the emperor but was largely ceremonial . Du Fu 's conscientiousness compelled him to try to make use[[ of]]
  - +4.27  in the London Borough of Hammersmith and Fulham . Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Black[[burn]]
  - +4.07  brings a touching vulnerability to the stage as William . "    Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Black[[burn]]
  - +4.04  sports shows and was eventually inducted into the Buffalo Broadcasters Hall of Fame .    In college , he led the Big Ten Conference in single season receptions during his senior year and set Michigan Wolverines receptions records for both career touch[[down]]
  - +3.09  The Ise @-@ class ships ' waterline protective belt had a maximum thickness of 299 mm ( 11 @.@ 8 in ) of Vickers cemented armour amidships ; below it was a stra[[ke]]
  - +2.85  review of the production for The Daily Telegraph , theatre critic Charles Spencer noted , " Robert Boulter brings a touching vulnerability to the stage as William . "    Boulter starred in two films in 2008 , Day[[light]]
  - +2.81 oyang , the region of his birthplace , was recovered by government forces in the winter of 76 , and in the spring of 765 Du Fu and his family sailed down the Yangtze , apparently with the intention of making their[[ way]]
Positive pole, set B:
  - +3.83  the war along with the remainder of the fleet to support the flight of SMS Goeben and SMS Breslau . The two German ships were attempting to break out of Messina , which was surrounded by British troops , and make their[[ way]]
  - +3.19 s official biography . " They also added elements like undistorted guitars and toned @-@ down vocal styles not heard on previous albums . "    King has since been critical of his performance , which he describes as his " most lackl[[uster]]
  - +3.10  of the fleet to support the flight of SMS Goeben and SMS Breslau . The two German ships were attempting to break out of Messina , which was surrounded by British troops , and make their way to Turkey . The break[[out]]
  - +2.94  processions , when the statue was carried out of the temple but still enclosed in a portable shrine . People did have less direct means of interaction . The more public parts of temples often incorporated small places for prayer , from doorways to freest[[anding]]
  - +2.93 0 pilots were trained . During World War II civil aerodromes were taken over for military use , existing military airfields were expanded , and new ones were built . This resulted in a significant inventory of facilities becoming available after the war . Pre[[ @]]
Negative pole, set A:
  - -1.99  from his performance as Trevor Nunn 's Hamlet ) , Robert Boulter , Shane Zaza and Fraser Ayres . " The Guardian noted , " Ben Whishaw and Robert Boulter offer tenderness amid the savag[[ery]]
  - -1.82  the ninth series of The X Factor UK on 10 December 2012 . According to the Daily Mail , their " energetic rendition " of " Kiss You " proved that the group have an elusive quality . On 1[[2]]
  - -1.81  Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . Boulter starred as " Darren " , in the 2[[0]]
  - -1.80 006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill . He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2[[0]]
  - -1.79  and had a role as a different character " Toby Steele " on The Bill .    He had a recurring role in 2003 on two episodes of The Bill , as character " Connor Price " . In 2[[0]]
  - -1.78  Direction 's previous hit singles , " What Makes You Beautiful " , " One Thing " , and " Live While We 're Young " . In April 2012 , The Independent reported that Simon Cowell , the group[[ ']]
  - -1.76  . By 18 January 2013 , the song had not been officially promoted to US radio stations . The track , however , was released by Sony Music Entertainment on 8 February 2013 , as the record[[ ']]
  - -1.74  's Arthur Hill High School in football , basketball , and track and . In 1943 , Michigan canceled boys high school tournaments in all sports due to World War II , and they did not return until the fall of 1[[9]]
Negative pole, set B:
  - -2.98  1918 . Two days later , Erzherzog Ferdinand Max and her sisters arrived in the port and assisted with the suppression of the mut . Following the restoration of order in the naval base , the armored cruisers S[[ank]]
  - -2.85  stations as well as 74 rounds of 190 mm shells aimed at Italian gun @-@ batteries and other port installations .    A major mutiny among crews of the armored cruisers stationed in Cattaro , including S[[ank]]
  - -2.67  . " Michael Roberts of Westworld Online said this was due to some of the numbers moving " at the sludgier of Black Sabbath . " Araya commented that the " album was a late bloomer — it wasn 't[[ really]]
  - -2.18  and light @-@ producing , and Osiris , who is static and shrouded in darkness , merge into a single god each night . Groups of three are linked with plurality in ancient Egyptian thought , and groups of four connote completeness . Rul[[ers]]
  - -2.09  back cover of South of Heaven around the time of 1986 's Reign in Blood . Lombardo felt it made Slayer seem as though they " had matured a little bit " , while Friedman himself deemed it " a[[ really]]
Writes up (raises the score of): [' and', ',', '-', ' of', ' (', '.', ' in', ' or', ':', ' -', ';', ' I']
Writes down (lowers the score of): ['Intf', 'ChannelType', 'MetaInfo', 'CreateModel', 'firehose', ' strugg', ' affili', 'uthors', '<filename>', 'mercial', 'vantages', 'KeyNot']
Zeroing it: mean loss change +0.0034 nats. Tokens hurt most:
  - +2.16 fer Girl " , Elvis Presley 's " Blue Hawaii " and Rammstein 's " Mein Land " , among others .    The music video garnered 10 @.@ 4 million views in a 4[[ @]]
  - +1.75  firing cycle of the main guns .    By World War II , the guns used Type 91 armour @-@ piercing , capped shells . Each of these shells weighed 673 @.@ 5 kilograms ( 1 @[[,]]
  - +1.40 2 @,@ 000 long tons ) to 40 @,@ 444 tonnes ( 39 @,@ 805 long tons ) . This also reduced the draught to 9 @[[.]]
  - +1.35 10 @.@ 4 million views in a 4 @-@ hour period , failing to attain the Vevo record held by Justin Bieber 's " Beauty and a Beat " music video ( 10 @[[.]]
  - +1.22  a better man , as a moral agent and as a perceiving organism " .    = = = Influence on Japanese literature = = =    Du Fu 's poetry has made a profound impact on Japanese literature , especially on the literature from the Mu[[rom]]
  - +1.22 @ 1 in ) to 2 @.@ 81 metres (  ft 3 in ) at full load as a result of the reduction in the displacement by over 2 @,@ 000 tonnes ( [[2]]

## O11 (carries part of T9)

Positive pole, set A:
  - +2.16 h plays such as Hyakuman , Bashō , and Shunkan .    During the Kan 'ei era of the Edo period ( 1624 – 1643 ) , Shào Chuán ( �[[�]]
  - +0.97  1911 , were outclassed by their British counterparts , the Eight @-@ Eight Fleet Program was restarted .    The first battleships built for the renewed Eight @-@ Eight Fleet Program were the two dreadn[[ough]]
  - +0.72 iser Ōyodo on arrival there the following day . The division continued onwards to Lingga . Its planned return to Japan was delayed by attacks by the American Third Fleet on targets in Indochina and southern China that sank two oil tank[[ers]]
  - +0.55  ) high @-@ angle guns had a maximum elevation of + 75 degrees , and had a rate of fire of 13 to 20 rounds per minute . They fired a 6 kg ( 13 lb )[[ projectile]]
  - +0.53  so often a great offensive end comes along , a player who has to learn how to play defense , but who has the natural speed , smooth actions , height and big hands that is the mark of an outstanding pass receiver . Rifenburg has laid[[ claim]]
  - +0.52  depicts the band shooting different scenes via a green screen , dressed as sailors , surfers , skiers and jailers . The video features scenes reminiscent of the films South Pacific , To Catch a Thief , Jailhouse Rock and Beach Blank[[et]]
  - +0.48  =    The design of the Fusō @-@ class battleships was shaped both by the ongoing international naval arms race and a desire among Japanese naval planners to maintain a fleet of capital ships powerful enough to defeat the United States Navy in an[[ encounter]]
  - +0.47  almost two years from late spring 766 . This period was Du Fu 's last great poetic flowering , and here he wrote 400 poems in his dense , late style . In autumn 766 , Bo Ma[[olin]]
Positive pole, set B:
  - +1.64 @ sign him following the 1993 – 94 season , he returned to the Penguins for one season before Tony Esposito convinced him to sign with the Tampa Bay Lightning in 1995 .    Cull[[en]]
  - +1.61  's all @-@ time scoring leader with 241 points , and was named to BU 's Hockey East 25th anniversary team in 2009 .    Passed over in the Entry Draft , Cull[[en]]
  - +1.52 8 – 99 season . He played his first game in nearly 18 months on September 18 , 198 , in an exhibition game between the Lightning and Sabres at Innsbruck , Austria . Cull[[en]]
  - +1.51  additional 50 @-@ games to injury , Cullen 's playing time and production declined .    The Penguins ' needs led them to complete a blockbuster trade on March 1 , 1991 . Cull[[en]]
  - +1.51  the bench in disbelief over how he was given a second chance . He was named to the roster and was greeted with a loud standing ovation by the fans in Tampa Bay when he was introduced prior to their season opening game .    Cull[[en]]
Negative pole, set A:
  - -2.03  he left the city to escape a rebellion , but he returned in summer 764 when he was appointed an advisor to Yan , who was involved in campaigns against the Tibetan Empire .    = = = Last years = = =    Lu[[oy]]
  - -1.96  life comes from his poems . His paternal grandfather was Du Shenyan , a noted politician and poet during the reign of Empress Wu . Du Fu was born in 712 ; the exact birthplace is unknown , except that it was near Lu[[oy]]
  - -1.93  Du Fu would have been allowed to enter the civil service because of his father 's rank , but he is thought to have given up the privilege in favour of one of his half brothers . He spent the next four years living in the Lu[[oy]]
  - -1.60  struck by five bombs and near missed multiple times ; all told she lost 50 crewmen killed and many others wounded . The bombs started numerous leaks and Ise began to settle by the bow , although she was returned to an even ke[[el]]
  - -1.14  in the Pacific Ocean : Britain and the United States . Satō Tetsutarō , a Japanese Navy admiral and military theorist , speculated that conflict would inevitably arise between Japan and at least one of its two main rivals . To that[[ end]]
  - -0.98  actually operated their aircraft in combat . While awaiting their air group the sister ships were sometimes used to ferry troops and material to Japanese bases . They participated in the Battle of Cape Engaño in late 1944 , where they dec[[oy]]
  - -0.96  its quality of production . Both Rolling Stone 's Jon Dolan , who praised its effectiveness , and Chris Payne of Billboard , who appreciated the melody , described " Kiss You " as one of the album 's highlights . Alexis Pet[[rid]]
  - -0.83  capital ships powerful enough to defeat the United States Navy in an encounter in Japanese territorial waters . The IJN 's fleet of battleships had proven highly successful in 1905 , the last year of the Russo @-[[@]]
Negative pole, set B:
  - -2.95  corporate aviation , and established the personal jet as a " whole new class of aircraft " .    = = Activities = =    The GA sector operates a range of aircraft , including balloons and airships , gliders , hang gliders , par[[ag]]
  - -1.77  deities , Egyptologist Claude Traunecker draws a distinction between " genies " — specialized patron spirits of certain places , objects , or activities , such as the sea or marsh god Wadj @-@ Wer and the harvest goddess Ren[[en]]
  - -1.27 ash / speed metal albums he heard in a long while " . He remarks , however , that " if you ’ re expecting to hear Reign in Blood Part Two , you ’ ll be in for a major disappointment " .    Kim Ne[[ely]]
  - -1.21  unknown and make vague , indirect references to other gods who are not even named . The Egyptologist James P. Allen estimates that more than 1 @,@ 400 deities are named in Egyptian texts , whereas his colleague Christian Le[[itz]]
  - -1.12  consistent with the elite tradition . The two traditions form a largely cohesive vision of the gods and their nature .    = = = Roles = = =    Most Egyptian deities represent natural or social phenomena . The gods were generally said to be imman[[ent]]
Writes up (raises the score of): ['iliency', 'makedirs', '<filename>', 'htaking', ' {:.', 'stanbul', 'qt', 'zep', 'eca', 'istical', 'з', 'asgow']
Writes down (lowers the score of): ['heres', 'born', 'ivari', '!”', 'ioid', ' —', '’', '”', ' unidentified', 'ryst', 'spring', '据']
Zeroing it: mean loss change -0.0000 nats. Tokens hurt most:
  - +0.34  @-@ millimetre ( 6 in ) secondary guns used in the Fusō class and earlier . To resolve this issue , the IJN designed a smaller 140 @-@ millimetre ( 5 @[[.]]
  - +0.32  = = = Armament = = =    The twelve 45 @-@ calibre 35 @.@ 6 cm ( 14 @.@ 0 in ) Type 41 guns of the I[[se]]
  - +0.27  located . This was just the first of a total of five airstrikes that the Americans launched that day . The sisters were not heavily engaged by the early airstrikes which are focusing on the group 's aircraft carriers . I[[se]]
  - +0.22  1943 and Hyūga at Sasebo Naval Arsenal from 2 May 30 November .    After completing her sea trials , Ise was attached to the Imperial Japanese Naval Academy at Etajima and fer[[ried]]
  - +0.20 10 @.@ 4 million views in a 4 @-@ hour period , failing to attain the Vevo record held by Justin Bieber 's " Beauty and a Beat " music video ( 10 @[[.]]
  - +0.18  on its quality of production . Both Rolling Stone 's Jon Dolan , who praised its effectiveness , and Chris Payne of Billboard , who appreciated the melody , described " Kiss You " as one of the album 's highlights . Alexis[[ Pet]]

## O890 (carries part of T9)

Positive pole, set A:
  - +1.16  was not completely suppressed for almost eight years . It caused enormous disruption to Chinese society : the census of 754 recorded 52 @.@ 9 million people , but ten years later , the census counted just 16[[ @]]
  - +1.13  tons ) . This also reduced the draught to 9 @.@ 03 metres ( 29 ft 8 in ) . The overhang of the flight deck at the stern increased the overall length to 219[[ @]]
  - +1.09  weight of the additional armour . In addition , their sterns were lengthened by 7 @.@ 62 metres ( 25 @.@ 0 ft ) . These changes increased their overall length to 213[[ @]]
  - +1.07 95 ft 2 in ) wide at its forward end and 13 metres ( 42 ft 8 in ) at the stern . It overhung the stern and increased the overall length of the ships to 219[[ @]]
  - +1.04  ( 19 ft 8 in ) high and designed to stow nine aircraft . It was fitted with fire fighting foam and carbon dioxide dispensers as a result of wartime experience . The ' T ' -shaped lift was 12[[ @]]
  - +1.03  ) . The overhang of the flight deck at the stern increased the overall length to 219 @.@ 62 metres ( 720 ft 6 in ) and the beam was slightly reduced to 31[[ @]]
  - +0.99  turret weighed 655 long tons ( 666 t ) . The hydraulically powered turrets had an elevation capability of − 5 / + 20 degrees . The guns had a rate of fire of 1[[ @]]
  - +0.96  rotating gunpowder @-@ propelled catapults were fitted on the sides of the hull , forward of the aft superstructure where they partially restricted the arc of fire of the two amidships turrets . They could launch aircraft up to 4[[ @]]
Positive pole, set B:
  - +0.98  interact with those in the sky . The Duat , in contrast , is treated as a remote and inaccessible place , and the gods who dwell there have difficulty communicating with those in the world of the living . The space outside the cosmos is also[[ said]]
  - +0.98  individuals have succeeded in international competition .    Of the 21 @,@ 000 civil aircraft registered in the UK , 96 per cent are engaged in GA operations , and annually the GA fleet accounts for between 1[[ @]]
  - +0.92  farm strips , over 80 per cent of GA activity is conducted at 134 of the larger aerodromes . The GA industry , which is around 7 per cent the size of its CAT cousin , employs 12[[ @]]
  - +0.92  — and demons , who have a more dangerous character . Many demons are hostile , causing illness and other troubles among humans . Their power can also be protective ; they may guard certain places in the Duat , the realm of the dead , or[[ advise]]
  - +0.90  is mostly drawn from religious writings produced by the nation 's scribes and priests . These people were the elite of Egyptian society and were very distinct from the general populace , most of whom were illiterate . Little is known about how well this broader population[[ knew]]
Negative pole, set A:
  - -1.09  ) , as well as in Australia , Canada , Denmark , France , New Zealand , and the Netherlands . The single peaked at number 46 on the US Billboard Hot 100 and has been certified gold by the Recording Industry[[ Association]]
  - -1.08 7 @,@ 000 digital downloads in the US by 18 January 2013 . The single ultimately peaked at number 46 on the Hot 100 and was certified gold by the Recording Industry[[ Association]]
  - -1.03  number 13 on the Australian Singles Chart on 27 January 2013 , marking its peak position and the group 's fourth top twenty hit in Australia . The song has been certified platinum by the Australian Recording Industry[[ Association]]
  - -0.98  It peaked at number 13 in its third and fourth charting weeks , becominh the group 's sixth top @-@ forty appearance in New Zealand . " Kiss You " has received a gold certification from the Recording Industry[[ Association]]
  - -0.94 34 other bombs near missed her , spraying her with splinters and ruptured some hull plates that contaminated some fuel oil and caused leaks in her port boiler rooms . While an exact total of her casualties is not available , it has been estimated[[ that]]
  - -0.90  The An Lushan Rebellion began in December 755 , and was not completely suppressed for almost eight years . It caused enormous disruption to Chinese society : the census of 754 recorded 52 @.@ 9[[ million]]
  - -0.89  its focus to the two remaining rivals for imperial dominance in the Pacific Ocean : Britain and the United States . Satō Tetsutarō , a Japanese Navy admiral and military theorist , speculated that conflict would inevitably arise between Japan and at[[ least]]
  - -0.76  almost eight years . It caused enormous disruption to Chinese society : the census of 754 recorded 52 @.@ 9 million people , but ten years later , the census counted just 16 @.@ 9[[ million]]
Negative pole, set B:
  - -1.55  might be labeled as deities are difficult to count . Egyptian texts list the names of many deities whose nature is unknown and make vague , indirect references to other gods who are not even named . The Egyptologist James P. Allen estimates that more than[[ ]]
  - -1.36 80 per cent of new pilots coming from the GA sector . The CAA estimates that between 1996 and 2006 the number of new professional pilots following the unsponsored training route rose from 48 per[[ cent]]
  - -1.27  such as racing , aerobatics , and parachuting , at which British teams and individuals have succeeded in international competition .    Of the 21 @,@ 000 civil aircraft registered in the UK , 96 per[[ cent]]
  - -1.15  aircraft movements . The pilot training element is regarded by the GA community as a key benefit that is critical to the supply of pilots for the airline industry . It is claimed by the General Aviation Awareness Council that 60 – 70 per[[ cent]]
  - -1.12  pilots coming from the GA sector . The CAA estimates that between 1996 and 2006 the number of new professional pilots following the unsponsored training route rose from 48 per cent to 59 per[[ cent]]
Writes up (raises the score of): ['DateTimeField', 'tidal', 'ebted', 'netes', 'adam', 'eas', 'nces', 'BLE', 'pared', 'itudinal', 'vents', 'rem']
Writes down (lowers the score of): [' per', 'Object', ' thickening', 'ual', 'oub', 'orney', 'terson', ' outl', 'ounce', ' II', 'joice', 'arian']
Zeroing it: mean loss change +0.0001 nats. Tokens hurt most:
  - +0.23 011 film Mercenaries directed by Paris Leonti .    = = Filmography = =    = = = Film = = =    = = = Television = = =    = = = Theatre = = =    = Du[[ Fu]]
  - +0.21  traveled slowly , held up by his ill @-@ health ( by this time he was suffering from poor eyesight , deafness and general old age in addition to his previous ailments ) . They stayed in Kuizhou ( in what is now B[[aid]]
  - +0.19  on Turret No. 5 for a Yokosuka E1Y reconnaissance floatplane in 1928 – 29 . A catapult and a collapsible 4 @- tonne ( 3 @[[.]]
  - +0.16 @ 5 kilograms ( 1 @,@ 485 lb ) and was fired at a muzzle velocity of 770 – 775 metres per second ( 2 @,@ 530 – [[2]]
  - +0.14  a designed output of 80 @,@ 000 shp ( 60 @,@ 000 kW ) intended to increase their speed to 24 @.@ 5 knots ( 45[[ @]]
  - +0.13 @ 0 ft ) . These changes increased their overall length to 213 @.@ 8 metres ( 701 ft ) , their beam to 31 @.@ 75 metres ( 10[[4]]

## O1486 (carries part of T9)

Positive pole, set A:
  - +0.81  of Empress Wu . Du Fu was born in 712 ; the exact birthplace is unknown , except that it was near Luoyang , Henan province ( Gong county is a favourite candidate ) . In later life , he considered[[ himself]]
  - +0.68  , who appreciated the melody , described " Kiss You " as one of the album 's highlights . Alexis Petridis for The Guardian commended the track 's chorus as " hard to dislodge from your brain " . Robert Cop[[sey]]
  - +0.56  again by American carrier aircraft in July . On the 24th Ise was struck by five bombs and near missed multiple times ; all told she lost 50 crewmen killed and many others wounded . The bombs started numerous leaks and[[ I]]
  - +0.56  , but the battleships were unable to find them , and Ozawa ordered them to reverse course and head for Amami Ōshima . When they arrived on 27 October , Ozawa transferred to Hyūga and hoisted[[ his]]
  - +0.48 6 during gunnery practice . Ise 's AA armament was upgraded in 1931 and Hyūga 's two years later . The latter ship was modernized in 1934 – 36 and[[ I]]
  - +0.48  recovered by government forces in the winter of 76 , and in the spring of 765 Du Fu and his family sailed down the Yangtze , apparently with the intention of making their way there . They traveled slowly , held up by[[ his]]
  - +0.42 5 . From 754 he began to have lung problems ( probably asthma ) , the first of a series of ailments which dogged him for the rest of his life . It was in that year that Du Fu was forced to move[[ his]]
  - +0.41  the Confucian classics of philosophy , history and poetry . He later claimed to have produced creditable poems by his early teens , but these have been lost .    In the early 730s , he travelled in the Jiangsu / Z[[he]]
Positive pole, set B:
  - +0.92  opening riff . It 's what we call a ' happy riff . ' It 's just like ' la @-@ lala @-@ la @-@ la @-@ la . ' I can 't see[[ myself]]
  - +0.68  Wallis Budge believed that Egyptian commoners were polytheistic , but knowledge of the true monotheistic nature of the religion was reserved for the elite , who wrote the wisdom literature . His contemporary James Henry Breasted thought Egyptian religion was instead pant[[he]]
  - +0.65  his speed and strength . The Lightning assigned him to the IHL 's Cleveland Lumberjacks , but also gave him the option of retiring and taking up a position as an assistant coach . He chose to accept the demotion , giving[[ himself]]
  - +0.60  term for whichever deity the reader chooses to revere . Although the combinations , manifestations , and iconographies of each god were constantly shifting , they were always restricted to a finite number of forms , never becoming fully interchangeable in a monotheistic or pant[[he]]
  - +0.55  his father 's old team , but injuries reduced his ability to perform . His most significant injury was a herniated disc in his neck that doctors initially feared would end his career . A bulky neck brace allowed Cullen to return and play out[[ his]]
Negative pole, set A:
  - -1.41  200 ft / s ) to a maximum height of 7 @,@ 500 metres ( 24 @,@ 600 ft ) . The ships were also fitted with six submerged 53[[ @]]
  - -1.35  minute . They fired a 6 kg ( 13 lb ) projectile with a muzzle velocity of 680 m / s ( 2 @,@ 200 ft / s ) to a maximum height of 7[[ @]]
  - -1.34 31 @.@ 75 metres ( 104 ft 2 in ) and their draft to 9 @.@ 45 metres ( 31 ft 0 in ) . Their displacement increased over 5[[ @]]
  - -1.31  gun mounts . When firing at surface targets , the guns had a range of 14 @,@ 700 metres ( 16 @,@ 100 yd ) ; they had a ceiling of 9[[ @]]
  - -1.30 @ 3 @-@ centimetre ( 21 @.@ 0 in ) torpedo tubes , three on each broadside . They carried twelve to eighteen 6th Year Type torpedoes which had a 200[[ @]]
  - -1.27 4 @,@ 681 t ) of coal and 1 @,@ 411 long tons ( 1 @,@ 434 t ) of fuel oil , which gave them a range of 9[[ @]]
  - -1.26  tons ) . This also reduced the draught to 9 @.@ 03 metres ( 29 ft 8 in ) . The overhang of the flight deck at the stern increased the overall length to 219[[ @]]
  - -1.25 .@ 07 mph ) from 81 @,@ 050 shp ( 60 @,@ 440 kW ) . The fuel storage of the ships was increased to a total of 5[[ @]]
Negative pole, set B:
  - -1.08  ) Vickers anti @-@ aircraft guns on the ship bought from Britain in 1910 and mounted onto Erzherzog Karl . Erzherzog Ferdinand Max was also fitted with two above water 45[[ @]]
  - -1.00  replica of the British 24 cm / 40 ( 9 @.@ 4 " ) Krupp C / 94 , which was used on the Habsburgs . Her secondary armament consisted of twelve 19[[ @]]
  - -0.98 ,@ 000 tonnes ( 1 @,@ 968 long tons ) . She was propelled by two two @-@ shaft , four cylinder vertical triple expansion steam engines . On trials , they developed 18[[ @]]
  - -0.96 22 @,@ 000 yd ) at maximum elevation with a muzzle velocity of 800 metres per second ( 2 @,@ 600 ft / s ) . The gun weighed 12[[ @]]
  - -0.96  @.@ 6 centimetres ( 2 @.@ 6 in ) / 45 caliber gun , also manufactured by Škoda . Anti @-@ aircraft and airship protection was covered by the four 37[[ @]]
Writes up (raises the score of): ['firehose', ' accred', ' strugg', ' interpre', ' assum', 'flation', ' impover', ' ardu', ' abre', ' necessit', ' coerc', ' incorpor']
Writes down (lowers the score of): ['ing', 'ater', ' and', ' for', 'Ke', 'e', ' For', ',', 'ed', ' or', 'ving', 'ara']
Zeroing it: mean loss change -0.0000 nats. Tokens hurt most:
  - +0.49 @ 000 ft ) at an elevation of 85 degrees . The maximum effective rate of fire was only between 10 and 120 rounds per minute because of the frequent need to change the fifteen @-@[[ round]]
  - +0.34 1 @.@ 75 metres ( 104 ft 2 in ) and their draft to 9 @.@ 45 metres ( 31 ft 0 in ) . Their displacement increased over 5 @[[,]]
  - +0.32  were replaced by four geared Kampon turbines with a designed output of 80 @,@ 000 shp ( 60 @,@ 000 kW ) intended to increase their speed to 24 @[[.]]
  - +0.26  ) in thickness and some of the armour removed from the turrets was used to protect the latter . The double bottom below the former positions of aft turrets was converted to hold fuel oil ; this increased the ships ' endurance to 9 @[[,]]
  - +0.20 04 ft 2 in ) and their draft to 9 @.@ 45 metres ( 31 ft 0 in ) . Their displacement increased over 5 @,@ 000 long tons ( 5[[ @]]
  - +0.19  forward end and 13 metres ( 42 ft 8 in ) at the stern . It overhung the stern and increased the overall length of the ships to 219 @.@ 62 metres ( 7[[2]]

## O83 (carries part of T9)

Positive pole, set A:
  - +0.53 0 due to digital download sales from its parent album . As a result of an " end @-@ of @-@ year download rush " on the week ending 30 December 2012 , the track re @-[[@]]
  - +0.52  rush " on the week ending 30 December 2012 , the track re @-@ entered the Hot 100 at number 83 . After the accompanying music video was released , the song re @-[[@]]
  - +0.51  the start of the Pacific War , both ships frequently exercised off the coasts of the Soviet Union , Korea and China in addition to training in Japanese waters . Ise hosted Edward , Prince of Wales , and his aide @-@ de @-[[@]]
  - +0.51  ) with the Detroit Lions for one season in 1950 . After retiring from football he settled in Buffalo and became a sports broadcaster . He worked as a color commentator and as a play @-@ by @-[[@]]
  - +0.50  set the stage for every subsequent writer in the genre .    In the 20th century , he was the favourite poet of Kenneth Rexroth , who has described him as " the greatest non @-@ epic , non @-[[@]]
  - +0.50  " a lot of hard work " by Payne , as " pure stupidity " by Louis Tomlinson , and as " I wouldn 't say [ it 's ] comedy , it 's all tongue @-@ in @-[[@]]
  - +0.49  , Du Fu is best known for his lǜshi , a type of poem with strict constraints on form and content , for example :    About two thirds of Du Fu 's 1500 extant works are in this[[ form]]
  - +0.49  praised the song for its production , calling it a stand @-@ out track on Take Me Home .    The track became the group 's sixth top @-@ ten hit in Ireland and the United Kingdom , while attaining top @-[[@]]
Positive pole, set B:
  - +0.61 ptica on a live medley of the two tracks at 1998 's Headbanger 's Heaven festival in the Netherlands . Adrien Begrand of PopMatters described " South of Heaven " as " an un[[orth]]
  - +0.54  aerial advertisement , etc . "    Organisations in the United Kingdom ( UK ) describe GA in less restrictive terms that include elements of commercial aviation . The British Business and General Aviation Association interprets it to be " all aeroplane and helicopter flying[[ except]]
  - +0.52  his illness . The team took an x @-@ ray and found a large black shadow in his chest . He underwent a CAT scan which revealed Cullen had a baseball @-@ sized tumor ; he was diagnosed as having Non @-[[@]]
  - +0.51  . [ Laughs ] I hate the opening riff . It 's what we call a ' happy riff . ' It 's just like ' la @-@ lala @-@ la @-@ la @-[[@]]
  - +0.51  Amun , to form Amun @-@ Ra ; with the solar form of Horus to form Ra @-@ Horakhty ; and with several solar deities as Horemakhet @-@ Khepri @-[[@]]
Negative pole, set A:
  - -1.42  = Background and release = =    " Kiss You " was written by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback , and its producers , Carl Falk and R[[ami]]
  - -1.37  on 7 January 2013 . The song was composed by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback and its producers , Carl Falk and R[[ami]]
  - -1.27 als    Niall Horan — additional guitar    Savan Kotecha — writing , background vocals    Kristian Lundin — writing    Albin Nedler — writing , background vocals    Shellback — writing    R[[ami]]
  - -1.26 ourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . In a review of the production for The Daily Telegraph , theatre critic Charles Spencer noted , " Robert Boulter brings a[[ touching]]
  - -1.23 .@ 71 metres ( 104 ft 0 in ) .    = = Ships = =    = = Service = =    Upon commissioning , the sister ships were assigned to the 1st Battleship Division of the[[ ]]
  - -1.21  room . While under repair at Kure , both ships received prototype Type 21 radars . Commanded by Vice @-@ Admiral Shirō Takasu , the 2nd Battleship Division set sail with the Aleutian Support[[ Group]]
  - -1.17  and of all the intonational potentials of the individual word , qualities that no translation can ever reveal . "    In his lifetime and immediately following his death , Du Fu was not greatly appreciated . In part this can be attributed to his stylistic[[ and]]
  - -1.16  displayed only singly " . He mastered all the forms of Chinese poetry : Chou says that in every form he " either made outstanding advances or contributed outstanding examples " . Furthermore , his poems use a wide range of registers , from the direct[[ and]]
Negative pole, set B:
  - -2.00  could write " slow material as well . " Metal Forces reviewer gives " the band credit for at least making an effort to try something new and not being afraid to experiment at such a crucial stage of their career " , creating " one of the[[ more]]
  - -1.18 o wing in the 1950s fostered the development of hang @-@ gliding during the 1960s and 1970s . The 1960s also saw experiments with motorised[[ hang]]
  - -1.15  creativity , gives form to all things by envisioning and naming them ; Atum produces all things as emanations of himself ; and Amun , according to the myths promoted by his priesthood , preceded and created the other creator gods . These and[[ other]]
  - -1.14  the 6 @.@ 6 centimetres ( 2 @.@ 6 in ) / 45 caliber gun , also manufactured by Škoda . Anti @-@ aircraft and airship protection was covered by the four[[ ]]
  - -1.13  saying “ John Cullen ... beat cancer and came back to play and helped us win . ”    = = Off the ice = =    Cullen and his wife Valerie have three daughters , Kennedy and twins Karlyn and Kort[[land]]
Writes up (raises the score of): ['ikipedia', 'ortunately', 'peutic', 'MetaInfo', 'prudence', 'everal', 'ulumi', 'wargs', 'ailure', 'ceptives', 'ccording', 'queeze']
Writes down (lowers the score of): [' in', '.', '3', '2', '4', '1', '0', 'ot', 'j', ' ', ' natural', '9']
Zeroing it: mean loss change +0.0003 nats. Tokens hurt most:
  - +0.38  a character named " Sean " in Donkey Punch , who tags along with character " Josh " as the " quiet brother ... who hits it off with Tammi " . Boulter guest starred on a two @-@[[ part]]
  - +0.33  Direction performed the track on The Today Show at the Rockefeller Center on 13 November 2012 , to a record crowd estimated at 15 @,@ 000 . " Kiss You " was included in the[[ set]]
  - +0.32  at any angle between -3 and + 20 degrees . In 1921 the elevation was increased to + 30 degrees and then to + 43 degrees during their mid @-@ 1930[[s]]
  - +0.26  on 7 January 2013 . The song was composed by Kristoffer Fogelmark , Kristian Lundin , Albin Nedler , Savan Kotecha , Shellback and its producers , Carl Falk and R[[ami]]
  - +0.26  were replaced by four geared Kampon turbines with a designed output of 80 @,@ 000 shp ( 60 @,@ 000 kW ) intended to increase their speed to 24 @[[.]]
  - +0.25     Niall Horan — additional guitar    Savan Kotecha — writing , background vocals    Kristian Lundin — writing    Albin Nedler — writing , background vocals    Shellback — writing    Rami[[ Y]]
