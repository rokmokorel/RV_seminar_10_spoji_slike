# import zahtevanih modulov
import numpy as np
import imutils
import cv2

class Zlepek:

	def __init__(self, slike):
		self.slike = slike
		self.razmerje = 0.75
		self.trash_projekcija = 4.0
		self.prikazi_ujemanja = False

	def spoji(self):
		return self.spoji_rekurzija(self.slike)

	def spoji_rekurzija(self, slike):
		# rekurzivno spajanje slik
		# poljubno stevilo zaporednih slik
		if len(slike) == 2:
			return self.spoji_sliki(slike[0], slike[1])
		for slika in slike[::-1]:
			return self.spoji_sliki(self.spoji_rekurzija(slike[:-1]), slika)

	def spoji_sliki(self, slikaL, slikaD):
		# lokalne znacilnosti in invariantni deskriptorji
		(slikaB, slikaA) = slikaL, slikaD
		(kpsA, znacilkaA) = self.zaznaj_in_opisi(slikaA)
		(kpsB, znacilkaB) = self.zaznaj_in_opisi(slikaB)

		# najdi ujemanja med deskriptorji
		M = self.poravnaj_znacilke(kpsA, kpsB, znacilkaA, znacilkaB)

		# ce je ujemanje None, ni dovoljsnega stevila ujemanj med slikama
		if M is None:
			return None

		# poenotenje perspektive nabora slik
		(ujemanja, H, status) = M
		rezultat = cv2.warpPerspective(slikaA, H,
			(slikaA.shape[1] + slikaB.shape[1], slikaA.shape[0]))
		rezultat[0:slikaB.shape[0], 0:slikaB.shape[1]] = slikaB
		rezultat = self.obrezi_vmesna(rezultat)

		# prikazi ujemanja glede na vrednost flaga
		if self.prikazi_ujemanja:
			vis = self.povezi_tocke(slikaA, slikaB, kpsA, kpsB, ujemanja,
				status)
			# vrnemo spojeno sliko in pa vizualizacijo ujemanja
			return (rezultat, vis)

		# vrnemo spojeno sliko
		return rezultat


	def zaznaj_in_opisi(self, slike):
		# pretvorba v sivinsko sliko
		sivinska = cv2.cvtColor(slike, cv2.COLOR_BGR2GRAY)
		# ustvarimo sift algoritem / scale-invariant feature transform		
		sift = cv2.xfeatures2d.SIFT_create()
		# shranimo keypoints in pa descriptorje keypoints
		(kps, znacilke) = sift.detectAndCompute(slike, None)
		# to array
		kps = np.float32([kp.pt for kp in kps])
		# vrnemo tuple
		return (kps, znacilke)

	def poravnaj_znacilke(self, kpsA, kpsB, znacilkaA, znacilkaB):
		# ustvarimo in zazenemo knn primerjevalnik, shranimo ujemanja v seznam
		primerjevalnik = cv2.DescriptorMatcher_create("BruteForce")
		vsa_ujemanja = primerjevalnik.knnMatch(znacilkaA, znacilkaB, 2)
		ujemanja = []

		# iteriramo po parih ujemajocih se tock
		for m in vsa_ujemanja:
			# preverimo dana ujemanja glede na kriterij 
			# Lowe's ratio test
			if len(m) == 2 and m[0].distance < m[1].distance * self.razmerje:
				ujemanja.append((m[0].trainIdx, m[0].queryIdx))

		# za dolocitev preslikava potrebujemo vsaj stiri tocke
		if len(ujemanja) > 4:
			# sestavimo mnozico tock A in B
			ptsA = np.float32([kpsA[i] for (_, i) in ujemanja])
			ptsB = np.float32([kpsB[i] for (i, _) in ujemanja])
			# poiscemo preslikavo med tockami
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				self.trash_projekcija)
			# vrnemo matriko ujemanja in preslikave
			return (ujemanja, H, status)
		# ce ni zadosti ujemanj, preslikave ni mogoce dolociti
		return None

	def povezi_tocke(self, slikaA, slikaB, kpsA, kpsB, ujemanja, status):
		# pripravino matriko izhodne slike
		(hA, wA) = slikaA.shape[:2]
		(hB, wB) = slikaB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = slikaA
		vis[0:hB, wA:] = slikaB
		# iteriramo skozi najdene pare tock
		for ((trainIdx, queryIdx), s) in zip(ujemanja, status):
			# ce je bila preslikava uspesna tocki izrisemo
			if s == 1:
				# narisemo ujemanje
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
		# vrnemo izhodno sliko
		return vis

	def obrezi_vmesna(self, in_slika):
		in_slika_grey = cv2.cvtColor(in_slika, cv2.COLOR_BGR2GRAY)
		# ustvarimo masko slike
		thresh = cv2.threshold(in_slika_grey, 0, 255, cv2.THRESH_BINARY)[1]
		obroba = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, \
		cv2.CHAIN_APPROX_SIMPLE)
		# zajamemo ogljisca obrobe
		obroba = imutils.grab_contours(obroba)
		# vrednosti v array
		c = max(obroba, key=cv2.contourArea)
		(x, y, w, h) = cv2.boundingRect(c)
		return in_slika[0:y + h, 0:x + w]

	def obrezi_sliko(self, slika):
		in_slika = cv2.copyMakeBorder(slika, 10, 10, 10, 10, \
			cv2.BORDER_CONSTANT, (0, 0, 0))
		in_slika_grey = cv2.cvtColor(in_slika, cv2.COLOR_BGR2GRAY)
		# ustvarimo masko slike
		thresh = cv2.threshold(in_slika_grey, 0, 255, cv2.THRESH_BINARY)[1]
		# poiscemo obrobe
		obroba = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, \
			cv2.CHAIN_APPROX_SIMPLE)
		# zajamemo ogljisca obrobe
		obroba = imutils.grab_contours(obroba)
		# vrednosti v array
		c = max(obroba, key=cv2.contourArea)
		# ustvarimo masko obmocja znotraj slike
		mask = np.zeros(thresh.shape, dtype="uint8")
		(x, y, w, h) = cv2.boundingRect(c)
		mask = cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
		# kopije maske
		# prva za obmocje najmanjsega kvadrata slike
		# druga je stevec za iteracijo zmanjsevanja kvadrata
		minRect = mask.copy()
		sub = mask.copy()
		# iteriramo dokler niso vse vrednosti vecje od nic
		while cv2.countNonZero(sub) > 0:
			# zmanjsaj velikost maske nato odstej podrocja
			minRect = cv2.erode(minRect, None)
			sub = cv2.subtract(minRect, thresh)
		# rezultat je minRect, maska z najmanjsim obmocjem slike
		# poiscemo obrobo pripravljenega najmanjsega kvadrata
		obroba_prava = cv2.findContours(minRect.copy(), \
			cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		obroba_prava = imutils.grab_contours(obroba_prava)
		# v array
		c = max(obroba_prava, key=cv2.contourArea)
		# narisemo kvadrat
		(x, y, w, h) = cv2.boundingRect(c)
		# uporabimo vrednosti odmikov kvadrata za zmanjsanje obsega spojene slike
		out_slika = in_slika[y:y + h, x:x + w]
		return out_slika
