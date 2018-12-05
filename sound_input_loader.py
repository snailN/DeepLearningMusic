#from scipy.io.wavfile import read 
import librosa
import numpy as np

def load_wav_input_data():

	training_filepath = [(r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\1Chiquitita.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\2AkaneSoraChihayafuruED2.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\3honey and clover ii je taime329.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\4Backstreet Boys Ill Never Break Your Heart.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\5Aji Bijon Ghore Indrani Sen.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\6Runner Chhutechhe Hemanta Mukherjee.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\7Duranta Ghurnir Ei Legechhe Pakflv Hemanta Mukherjee.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\8Baek Ji Young That Girl Secret Garden OST.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\9siganui sup Every Single Day.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\10ddr 3 idiots behti hawa sa tha woh.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\11taare zameen par.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\12Mozart  ave maria voice.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\13brian adams bryan adams summer of 69.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\14Edith Piaf Non Je ne regrette rien.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\15Cali Pachanguero Grupo Niche.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\16Inst goong ice pond.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\17amclassical_march_from_the_magic_flute.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\18mozart eine kleine nachtmusik.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\19clarinet polka.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\20yawarakana jikan.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\21richard clyderman ballade pour adeline.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\22yiruma kiss the rain.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\23Lost Cause Instrumental.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\24suzuki violin method vol 03 track 5 gavotte j becke.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\25bach instrumental ave maria violin piano.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\26Guitar Bach Lute Suite in E minor Julian Bream_classical guitar 6 Gigue.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\27amclassical_pathetique_sonata_movement_2.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\28pachelbels canon in d major.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\29chopin gran bals brillante113.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\30Firebird Suite Finale.wav', 0)]


	validation_filepath = [(r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\ValiData\1mozart violin concerto no 3.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\ValiData\2John Marsh Symphony No6 in Dmajor.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\ValiData\3bon jovi its my life.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\ValiData\4AMI CHOLTE CHOLTE THEME GECHI.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\ValiData\5yotsuba no clover.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\ValiData\6Celine Dion Its All Coming Back To Me Now.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\ValiData\7Audiomachine  Final Hope.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\ValiData\8Jang Dong Gun More than me.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\ValiData\9Friendship Okazaki Ritsuko Sukitte_i Na Yo.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\ValiData\10Jattendrai Rina Ketty.wav', 1)]


	test_filepath = [(r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TestData\1Manna De Coffee Houser Shei Addata.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TestData\2chris de burg the lady in red.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TestData\3disney.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TestData\4len paganini caprice24.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TestData\5allegro cantable voice.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TestData\6Danity kane Stay With Me.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TestData\7MLTR Complicated Heart.wav', 1), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TestData\8ROSSINI William Tell Overture.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TestData\9the four seasons388.wav', 0), (r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TestData\10Franz Liszt La Campanella.wav', 0)]
 
    
	#librosa.util.example_audio_file()

	#a = read('dhitang dhitang bole.wav')

	#data, rate = librosa.load(filepath, sr=None)

	#print(a[0]) 
	#print(np.array(a[1],dtype=float)) 
	#print(np.array(a[1],dtype=float).shape) 

	#print(data)
	#print(rate)
	#print(data.shape)

	#print(len(training_filepath))

	wavdatalist = []
	wavlabellist = []

	#fn = [(r'C:\Users\parom\Desktop\Work\Python_Projects\InputDataRes\InstVoiceClassif\TrainData\1Chiquitita.wav', 1)]

	#for a, b in fn:
		#data, rate = librosa.load(a, sr=None)
		#print(data)
		#print(data.dtype)
		#print(data.shape)
			

	for path in training_filepath:
		data, rate = librosa.load(path[0], sr=None, offset = 20.0) 		
		wavdatalist.append(data)
		wavlabellist.append(path[1]) 
			

	#maxwavlen = max(wavdatalist)
	#for witem in wavdatalist:
		#if len(witem) == maxwavlen:
			#continue
		#else:
			#np.pad(witem...from maxwavlen index)
			
			
	#print(wavdatalist)
	#print(wavlabellist)

	wavdataarray = np.float64(np.vstack(wavdatalist))
	wavlabelarray = np.hstack(wavlabellist)

	#print(wavdataarray)
	#print(wavdataarray.shape)
	#print(wavlabelarray)
	#print(wavlabelarray.shape)

	trainingDataTup = (wavdataarray, wavlabelarray)
	print(trainingDataTup)

	#a = [list(a) for a in wavdatalist]

	#validation data

	validwavdatalist = []
	validwavlabellist = []

	for path in validation_filepath:
		data, rate = librosa.load(path[0], sr=None, offset = 20.0)     
		validwavdatalist.append(data)
		validwavlabellist.append(path[1])


	validwavdataarray = np.float64(np.vstack(validwavdatalist))
	validwavlabelarray = np.hstack(validwavlabellist)

	#print(validwavdataarray)
	#print(validwavdataarray.shape)
	#print(validwavlabelarray)
	#print(validwavlabelarray.shape)

	validationDataTup = (validwavdataarray, validwavlabelarray)
	print(validationDataTup)


	#test data

	testwavdatalist = []
	testwavlabellist = []

	for path in test_filepath:
		data, rate = librosa.load(path[0], sr=None, offset = 20.0)     
		testwavdatalist.append(data)
		testwavlabellist.append(path[1])


	testwavdataarray = np.float64(np.vstack(testwavdatalist))
	testwavlabelarray = np.hstack(testwavlabellist)

	#print(testwavdataarray)
	#print(testwavdataarray.shape)
	#print(testwavlabelarray)
	#print(testwavlabelarray.shape)

	testDataTup = (testwavdataarray, testwavlabelarray)
	print(testDataTup)
	
	return(trainingDataTup, validationDataTup, testDataTup)
	

def input_data_wrapper():
	
	tr_d, va_d, te_d = load_wav_input_data()
	
	training_inputs = [np.reshape(x, (441000, 1)) for x in tr_d[0]]
	training_results = [[[y]] for y in tr_d[1]]
	training_data = zip(training_inputs, training_results)
	
	validation_inputs = [np.reshape(x, (441000, 1)) for x in va_d[0]]	
	validation_data = zip(validation_inputs, va_d[1])
	
	test_inputs = [np.reshape(x, (441000, 1)) for x in te_d[0]]	
	test_data = zip(test_inputs, te_d[1])
	
	return(training_data, validation_data, test_data)
	